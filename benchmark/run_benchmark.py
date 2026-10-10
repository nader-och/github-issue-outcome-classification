"""
Score every model's test-set predictions the same way and build the benchmark table.

Run from the repository root, after the training scripts:
    py benchmark/run_benchmark.py

Input : results/predictions/*.csv   one file per model, columns id, label, pred, prob_not_planned
        data/cleaned/test.csv       used to check the ids and to get each issue's repository
Output: results/benchmark.csv       all metrics, one row per model
        results/benchmark.md        the same as a Markdown table for the README / slides

Primary metric: macro-F1 (the plain average of the F1 of each class). Both classes count
equally, so a model cannot score well by favouring the larger class, and it stays honest on
skewed repositories such as numpy (76% completed). Accuracy and ROC-AUC are reported too.

To add a model (role 4): drop its predictions file into results/predictions/ and add one line
to MODEL_INFO below. A file that is not in MODEL_INFO is still scored, with type "unknown".
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "predictions"
TEST_FILE = ROOT / "data" / "cleaned" / "test.csv"
OUT_CSV = ROOT / "results" / "benchmark.csv"
OUT_MD = ROOT / "results" / "benchmark.md"

N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# model file name -> (type, short description). "baseline" = taught in class in vanilla form,
# "new method" = not taught (announcement rules).
MODEL_INFO = {
    "majority":               ("baseline", "Always predicts the majority class (no learning)"),
    "naive_bayes":            ("baseline", "Multinomial Naive Bayes, word TF-IDF 1-2-grams"),
    "logreg_word":            ("baseline", "Logistic Regression, word TF-IDF 1-2-grams"),
    "logreg_word_char_stats": ("baseline", "Logistic Regression, word + char 3-5-grams + statistical features"),
    "linear_svm":             ("baseline", "Linear SVM (Platt-calibrated), word + char + statistical features"),
    # Role 4 adds its models here, e.g.
    # "glove_cnn":   ("baseline", "GloVe 100-d + CNN"),
    # "bilstm":      ("baseline", "GloVe 100-d + BiLSTM"),
    # "distilbert":  ("baseline", "Fine-tuned distilbert-base-uncased"),
    # "deberta_v3":  ("new method", "Fine-tuned microsoft/deberta-v3-base"),
}

REQUIRED_COLUMNS = ["id", "label", "pred", "prob_not_planned"]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred):
    """95% confidence interval of macro-F1: resample the test set with replacement
    N_BOOTSTRAP times. Shows how much the score could move with a different test sample,
    i.e. whether a gap between two models is bigger than the noise."""
    rng = np.random.default_rng(RANDOM_STATE)  # same resamples for every model
    n = len(y_true)
    scores = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        scores.append(macro_f1(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 97.5])


def load_predictions(path, test):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if len(df) != len(test) or set(df["id"]) != set(test["id"]):
        raise ValueError(f"{path.name}: ids do not match data/cleaned/test.csv "
                         f"({len(df)} rows vs {len(test)}); predictions must be on the test set only")
    df = df.merge(test[["id", "repo", "label"]], on="id", suffixes=("", "_true"))
    if (df["label"] != df["label_true"]).any():
        raise ValueError(f"{path.name}: the label column does not match test.csv")
    return df


def score(name, df, repos):
    y, pred = df["label"].to_numpy(), df["pred"].to_numpy()
    prob = df["prob_not_planned"]
    f1_each = f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    low, high = bootstrap_ci(y, pred)
    kind, description = MODEL_INFO.get(name, ("unknown", ""))
    row = {
        "model": name,
        "type": kind,
        "macro_f1": macro_f1(y, pred),
        "macro_f1_ci_low": low,
        "macro_f1_ci_high": high,
        "accuracy": accuracy_score(y, pred),
        "f1_completed": f1_each[0],
        "f1_not_planned": f1_each[1],
        # ROC-AUC only when the model gives real probabilities (not empty, not constant).
        "roc_auc": roc_auc_score(y, prob) if prob.notna().all() and prob.nunique() > 1 else np.nan,
    }
    for repo in repos:
        part = df[df["repo"] == repo]
        row[f"f1_{repo.split('/')[1]}"] = macro_f1(part["label"].to_numpy(), part["pred"].to_numpy())
    row["description"] = description
    return row


def to_markdown(table, repos):
    def fmt(x):
        return "–" if pd.isna(x) else f"{x:.3f}"

    lines = [
        "# Benchmark (test set, " + str(table.attrs["n_test"]) + " issues)",
        "",
        "Sorted by macro-F1 (primary metric). CI = 95% bootstrap confidence interval of macro-F1.",
        "",
        "| Rank | Model | Type | Macro-F1 | 95% CI | Accuracy | F1 completed | F1 not_planned | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(table.itertuples(), 1):
        lines.append(f"| {i} | `{r.model}` | {r.type} | **{fmt(r.macro_f1)}** | "
                     f"{fmt(r.macro_f1_ci_low)}–{fmt(r.macro_f1_ci_high)} | {fmt(r.accuracy)} | "
                     f"{fmt(r.f1_completed)} | {fmt(r.f1_not_planned)} | {fmt(r.roc_auc)} |")

    short = [r.split("/")[1] for r in repos]
    lines += ["", "## Macro-F1 per repository", "",
              "| Model | " + " | ".join(short) + " |",
              "|---|" + "---|" * len(short)]
    for _, r in table.iterrows():
        lines.append(f"| `{r['model']}` | " + " | ".join(fmt(r[f'f1_{s}']) for s in short) + " |")
    lines += ["", "numpy has only 115 test issues (27 not_planned), so its per-repository score is noisy.", ""]
    return "\n".join(lines)


def main():
    test = pd.read_csv(TEST_FILE)
    repos = sorted(test["repo"].unique())
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("No files in results/predictions/. Run the training scripts first.")

    rows = []
    for path in files:
        rows.append(score(path.stem, load_predictions(path, test), repos))
        print(f"scored {path.stem}")

    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    table.attrs["n_test"] = len(test)
    table.round(4).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(to_markdown(table, repos), encoding="utf-8")

    print()
    print(table[["model", "type", "macro_f1", "accuracy", "roc_auc"]].round(4).to_string(index=False))
    print(f"\nsaved {OUT_CSV.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
