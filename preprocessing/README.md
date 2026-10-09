# Preprocessing: cleaning and splitting

`clean.py` turns the raw issues in `data/raw/github_issues_raw.jsonl.gz` into clean, leak-free train / validation / test sets.

```bash
py preprocessing/clean.py        # Windows (python3 on macOS/Linux); runs in about 10 seconds
```

| Output | Content |
|---|---|
| `data/cleaned/train.csv` | 7,808 issues (80%) |
| `data/cleaned/val.csv` | 976 issues (10%) – for choosing hyperparameters and epochs |
| `data/cleaned/test.csv` | 977 issues (10%) – used once, for the final benchmark |
| `data/cleaned/cleaning_report.json` | row and class counts after every step, per-repo counts, median lengths, SHA-256 of each CSV |

CSV columns: `id` (`<repo>#<number>`), `repo`, `created_month` (`YYYY-MM`), `text`, `label`.

**Label encoding:** `label = 1` for `not_planned` (rejected), `label = 0` for `completed` (resolved).

The script is deterministic (`random_state = 42`, stable sorting, fixed line endings): running it again produces byte-identical files, which can be checked against the SHA-256 values in the report.

## Steps and counts

| # | Step | Rows | Removed | completed | not_planned |
|---|---|---:|---:|---:|---:|
| 0 | Raw file | 10,399 | – | 5,562 | 4,837 |
| 1 | Close reason still matches the label | 10,399 | 0 | 5,562 | 4,837 |
| 2 | Remove bot accounts | 10,390 | 9 | 5,554 | 4,836 |
| 3 | Remove CI automation accounts | 10,188 | 202 | 5,489 | 4,699 |
| 4–5 | Build and clean the text | 10,188 | 0 | 5,489 | 4,699 |
| 6 | Remove near-empty issues | 10,130 | 58 | 5,478 | 4,652 |
| 7a | Remove exact duplicate texts | 10,018 | 112 | 5,399 | 4,619 |
| 7b | Remove duplicate titles within a repo | 9,761 | 257 | 5,215 | 4,546 |

**Final dataset: 9,761 issues (53.4% completed, 46.6% not_planned).** Median cleaned length: 106 words (689 characters).

### 1. Consistent label
Keeps only issues whose `state_reason` at download time equals the class they were collected for. This would drop issues that were reopened or re-closed with a different reason between the search and the download. It removed nothing in this run, but it keeps the label trustworthy if the data is collected again.

### 2. Bot accounts
Removes issues whose author has `user_type == "Bot"` (e.g. `github-actions[bot]`, `vs-code-engineering[bot]`). Machine-written reports are not the task: we want to predict how maintainers react to issues written by people.

### 3. CI automation accounts
Spot-checking the most frequent authors of each repository found two accounts that post issues automatically from CI pipelines but are registered as normal users, so step 2 misses them:

- `typescript-bot` (163 issues, 80% `not_planned`): nightly `[ServerErrors]` / `[NewErrors]` reports.
- `fluttergithubbot` (39 issues, 82% `completed`): "*test X is 2.13% flaky*" reports.

Their texts are near-identical templates with a strongly skewed label, so a model would score points by memorising them instead of learning anything about real issues.

### 4. Build the text
`text = title + "\n\n" + body` (a missing body becomes an empty string). **Only the title and body are used.** `labels`, `comments`, `closed_at`, `state_reason` and `author_association` are never used as features, because they are set or change during triage and closing and would leak the answer. (`user_type` and `user_login` are used only to filter rows in steps 2–3, never as features.)

### 5. Clean the text
Issue templates add a lot of text that is the same in every issue, and code, logs and screenshots make texts very long without adding words a classifier can use. The cleaning, in order:

| What | Becomes | Why |
|---|---|---|
| HTML comments `<!-- ... -->` | removed | issue templates are full of hidden instructions |
| Collapsible `<details>` blocks | `[DETAILS]` | VS Code's issue reporter puts system info, extension lists and A/B experiment IDs there |
| Fenced code blocks ` ``` ` / `~~~` | `[CODE]` | |
| Images, `<img>`/`<video>` tags, uploaded attachments | `[IMAGE]` | |
| URLs (markdown links keep their text) | `[URL]` | |
| 3+ consecutive stack-trace or log lines outside code blocks | `[TRACE]` | JavaScript, Dart, Python frames, timestamped and Kubernetes log lines |
| HTML formatting tags (`<b>`, `<br>`, ...), inline-code backticks | removed (text kept) | |
| `_No response_`, unchecked checklist lines `- [ ]`, headings with nothing under them | removed | unanswered template fields |
| Extra spaces and blank lines | collapsed | |

Code, links and screenshots are **replaced by tokens instead of deleted**, so the information "this report contains code / a link / a screenshot / a log" survives (a report with a reproduction is probably more likely to be fixed) while the text stays short enough for transformer models. Share of training issues containing each token: `[CODE]` 54%, `[URL]` 53%, `[DETAILS]` 22%, `[IMAGE]` 16%, `[TRACE]` 0.7% (most logs are already inside code blocks). Letter case is kept; each model decides whether to lowercase.

### 6. Near-empty issues
Removes issues whose cleaned text is shorter than 20 characters (e.g. "Pad", "del", "Test Issue Ignore"): there is nothing to learn from them.

### 7. Duplicates
Duplicates are removed **before** splitting, so the same text can never appear in both train and test (that would be data leakage and inflate the scores). The earliest-created copy is kept, since later copies are usually re-reports of the same problem.

- **7a.** Exact duplicate cleaned texts (12 of these groups had copies with different labels).
- **7b.** Issues in the same repository whose titles are equal after lowercasing and replacing every number with `0`. This catches templated reports that differ only in a number, e.g. Flutter infrastructure tickets "*mac-10 lost external connection from phone device.*" / "*linux-46 is dead.*" (168 copies from one author, all `completed`) and TypeScript's weekly "*Design Meeting Notes, 3/5/2024*".

### 8–9. Label and split
The label is encoded as above and the data is split **80 / 10 / 10**, stratified on **repository + label together** (`random_state = 42`). Stratifying on both keeps the class balance of each repository the same in every split. This matters most for numpy, which is strongly skewed (875 completed vs 273 not_planned): with a plain random split, its 27 test `not_planned` issues could easily end up much fewer, and its per-repo score would be unreliable.

The script checks before saving that no `id` and no `text` appears in two splits, and that every split contains both classes for every repository.

| Split | Rows | completed | not_planned |
|---|---:|---:|---:|
| train | 7,808 | 4,172 | 3,636 |
| val | 976 | 521 | 455 |
| test | 977 | 522 | 455 |

Per-repository counts for every split are in `cleaning_report.json`.
