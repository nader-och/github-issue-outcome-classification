"""
Clean the raw GitHub issues and split them into train / validation / test sets.

Run from the repository root:
    py preprocessing/clean.py

Input : data/raw/github_issues_raw.jsonl.gz   (made by data/collect.py)
Output: data/cleaned/train.csv, val.csv, test.csv  (columns: id, repo, created_month, text, label)
        data/cleaned/cleaning_report.json          (row counts after every step)

Label encoding: 1 = not_planned (rejected), 0 = completed (resolved).

The model input is ONLY the issue title + body. Metadata such as labels, comments,
closed_at, state_reason or author_association is never used as a feature, because it
is set or changed during triage/closing and would leak the answer.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "github_issues_raw.jsonl.gz"
OUT_DIR = ROOT / "data" / "cleaned"

RANDOM_STATE = 42
MIN_TEXT_CHARS = 20
LABELS = {"completed": 0, "not_planned": 1}

# Accounts that post issues automatically from CI pipelines but are registered on
# GitHub as normal "User" accounts, so the user_type == "Bot" filter misses them.
# Found by spot-checking the most frequent authors of every repository.
AUTOMATION_ACCOUNTS = {
    "typescript-bot",    # microsoft/TypeScript: nightly "[ServerErrors]" / "[NewErrors]" reports
    "fluttergithubbot",  # flutter/flutter: "<test> is X% flaky" reports
}

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.DOTALL)
# Collapsible <details> blocks hold system info, extension lists and long logs.
DETAILS_BLOCK = re.compile(r"<details\b.*?(</details>|$)", re.DOTALL | re.IGNORECASE)
# Fenced code blocks (``` or ~~~). An unclosed fence runs to the end of the text.
CODE_FENCE = re.compile(r"(```|~~~).*?(\1|$)", re.DOTALL)
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_IMAGE = re.compile(r"<(img|video)\b[^>]*>", re.IGNORECASE)
# Bare links to uploaded screenshots / screen recordings.
ATTACHMENT_URL = re.compile(
    r"https?://\S*(user-attachments/assets|user-images\.githubusercontent\.com)\S*"
    r"|https?://\S+\.(png|jpe?g|gif|webp|mp4|mov)\b\S*",
    re.IGNORECASE,
)
MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")
URL = re.compile(r"https?://\S+|www\.\S+")
# Only real HTML formatting tags, so placeholders like <node-name> in commands survive.
HTML_TAG = re.compile(
    r"</?(a|b|i|u|s|em|strong|br|p|div|span|sub|sup|kbd|code|pre|center|picture|source|"
    r"summary|h[1-6]|ul|ol|li|table|thead|tbody|tr|td|th|hr|blockquote)\b[^>]*>",
    re.IGNORECASE,
)
NO_RESPONSE = re.compile(r"^\s*_No response_\s*$", re.MULTILINE)
UNCHECKED_BOX = re.compile(r"^\s*[-*]\s*\[ \].*$", re.MULTILINE)
HEADING = re.compile(r"^\s*#{1,6}\s")

# Lines that look like stack-trace frames or log output. A run of 3 or more
# such lines (outside code fences) is replaced by a single [TRACE] token.
TRACE_LINE = re.compile(
    r"^\s*("
    r"at \S+.*"                                   # JavaScript / Java / C#: "at foo (file.js:1:2)"
    r"|#\d+\s+\S+.*"                              # Dart / C++: "#0  main (file.dart:3)"
    r"|File \".*\", line \d+.*"                   # Python frame
    r"|Traceback \(most recent call last\):"      # Python header
    r"|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}.*"  # timestamped log line
    r"|[IWEF]\d{4} \d{2}:\d{2}:\d{2}.*"           # Kubernetes klog line
    r")$"
)


def replace_traces(text: str) -> str:
    out, run = [], []
    for line in text.split("\n"):
        if TRACE_LINE.match(line):
            run.append(line)
            continue
        out.extend(["[TRACE]"] if len(run) >= 3 else run)
        run = []
        out.append(line)
    out.extend(["[TRACE]"] if len(run) >= 3 else run)
    return "\n".join(out)


def drop_empty_headings(text: str) -> str:
    """Remove template headings that have no content under them."""
    lines = text.split("\n")
    keep = []
    for i, line in enumerate(lines):
        if HEADING.match(line):
            following = next((l for l in lines[i + 1:] if l.strip()), None)
            if following is None or HEADING.match(following):
                continue
        keep.append(line)
    return "\n".join(keep)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_COMMENT.sub(" ", text)
    text = DETAILS_BLOCK.sub(" [DETAILS] ", text)
    text = CODE_FENCE.sub(" [CODE] ", text)
    text = MD_IMAGE.sub(" [IMAGE] ", text)
    text = HTML_IMAGE.sub(" [IMAGE] ", text)
    text = ATTACHMENT_URL.sub(" [IMAGE] ", text)
    text = text.replace("`", "")            # inline code `x` -> x (short, so it is kept)
    text = MD_LINK.sub(r"\1 [URL]", text)   # keep the link text, mark that a link was there
    text = URL.sub(" [URL] ", text)
    text = HTML_TAG.sub(" ", text)
    text = replace_traces(text)
    text = NO_RESPONSE.sub("", text)
    text = UNCHECKED_BOX.sub("", text)
    text = drop_empty_headings(text)
    # Collapse whitespace: single spaces inside lines, at most one blank line.
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def log_step(report: dict, name: str, df: pd.DataFrame, note: str) -> None:
    before = report["steps"][-1]["rows"] if report["steps"] else len(df)
    report["steps"].append({
        "step": name,
        "rows": len(df),
        "removed": before - len(df),
        "completed": int((df["query_class"] == "completed").sum()),
        "not_planned": int((df["query_class"] == "not_planned").sum()),
        "note": note,
    })
    print(f"{name:<32} {len(df):>6} rows  (-{before - len(df)})")


def main() -> None:
    report = {"steps": []}

    # 1. Load; keep only issues whose current close reason still matches the label.
    df = pd.read_json(RAW_FILE, lines=True)
    log_step(report, "0_raw", df, "raw file as downloaded")
    df = df[df["state_reason"] == df["query_class"]]
    log_step(report, "1_consistent_label", df,
             "state_reason equals query_class (drops issues reopened or re-closed differently)")

    # 2. Remove issues written by bot accounts.
    df = df[df["user_type"] != "Bot"]
    log_step(report, "2_no_bots", df, 'user_type != "Bot"')

    # 3. Remove issues posted automatically by CI accounts that look like normal users.
    df = df[~df["user_login"].isin(AUTOMATION_ACCOUNTS)]
    log_step(report, "3_no_automation_accounts", df,
             "removed accounts: " + ", ".join(sorted(AUTOMATION_ACCOUNTS)))

    # 4-5. Build the model input from title + body only, then clean it.
    title = df["title"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    df = df.assign(text=(title + "\n\n" + body).map(clean_text))
    log_step(report, "4_5_build_and_clean_text", df, "text = clean(title + body); no rows removed")

    # 6. Remove near-empty issues.
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
    log_step(report, "6_no_near_empty", df, f"cleaned text has at least {MIN_TEXT_CHARS} characters")

    # 7. Remove duplicates, keeping the earliest-created copy (later copies are usually
    #    re-reports of the same problem). Titles are compared case-insensitively with all
    #    numbers masked, so templated reports such as "mac-10 is dead." / "linux-46 is dead."
    #    or "Design Meeting Notes, 3/5/2024" count as duplicates.
    df = df.sort_values(["created_at", "repo", "number"], kind="mergesort")
    conflicts = int(df[df.duplicated("text", keep=False)].groupby("text")["query_class"].nunique().gt(1).sum())
    df = df.drop_duplicates("text", keep="first")
    log_step(report, "7a_no_duplicate_text", df,
             f"exact duplicate cleaned text, keep earliest ({conflicts} duplicate groups had mixed labels)")
    title_key = (df["title"].fillna("").str.lower()
                 .str.replace(r"\d+", "0", regex=True)
                 .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df[~df.assign(title_key=title_key).duplicated(["repo", "title_key"], keep="first")]
    log_step(report, "7b_no_duplicate_title", df,
             "same repo + same title (lower-cased, numbers masked), keep earliest")

    # 8. Encode the label and build the output columns.
    df = pd.DataFrame({
        "id": df["repo"] + "#" + df["number"].astype(str),
        "repo": df["repo"],
        "created_month": pd.to_datetime(df["created_at"], utc=True).dt.strftime("%Y-%m"),
        "text": df["text"],
        "label": df["query_class"].map(LABELS).astype(int),
    }).sort_values("id", kind="mergesort").reset_index(drop=True)

    # 9. Split 80 / 10 / 10, stratified on repo + label together.
    strata = df["repo"] + "|" + df["label"].astype(str)
    train, rest = train_test_split(df, test_size=0.2, stratify=strata, random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.5, stratify=strata.loc[rest.index],
                                 random_state=RANDOM_STATE)
    splits = {"train": train, "val": val, "test": test}

    # Checks: no id or text shared between splits; both classes in every repo of every split.
    for a in splits:
        for b in splits:
            if a < b:
                assert not set(splits[a]["id"]) & set(splits[b]["id"]), f"id overlap {a}/{b}"
                assert not set(splits[a]["text"]) & set(splits[b]["text"]), f"text overlap {a}/{b}"
    for name, part in splits.items():
        per_repo = part.groupby("repo")["label"].nunique()
        assert len(per_repo) == df["repo"].nunique() and (per_repo == 2).all(), \
            f"{name}: a repo is missing a class"

    # 10. Save the CSVs and the report.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report["label_encoding"] = LABELS
    report["random_state"] = RANDOM_STATE
    report["splits"] = {}
    for name, part in splits.items():
        part = part.sort_values("id", kind="mergesort")
        path = OUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        words = part["text"].str.split().str.len()
        report["splits"][name] = {
            "rows": len(part),
            "completed": int((part["label"] == 0).sum()),
            "not_planned": int((part["label"] == 1).sum()),
            "median_chars": int(part["text"].str.len().median()),
            "median_words": int(words.median()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report["per_repo"] = {
        repo: {name: {"completed": int((p.loc[p["repo"] == repo, "label"] == 0).sum()),
                      "not_planned": int((p.loc[p["repo"] == repo, "label"] == 1).sum())}
               for name, p in splits.items()}
        for repo in sorted(df["repo"].unique())
    }
    report["median_chars_all"] = int(df["text"].str.len().median())
    report["median_words_all"] = int(df["text"].str.split().str.len().median())

    with open(OUT_DIR / "cleaning_report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print()
    for name, s in report["splits"].items():
        print(f"{name:<6} {s['rows']:>6} rows  completed={s['completed']}  not_planned={s['not_planned']}"
              f"  median_words={s['median_words']}")
    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
