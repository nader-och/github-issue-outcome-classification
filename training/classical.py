"""
Train and evaluate the classical (Week 2) baselines.

Run from the repository root:
    py training/classical.py                 # all models (a few minutes)
    py training/classical.py --cross-repo    # also the cross-repository test (a few minutes more)

Input : data/cleaned/train.csv, val.csv, test.csv  (made by preprocessing/clean.py)
Output: results/predictions/<model_name>.csv       test-set predictions in the shared format
                                                    (id, label, pred, prob_not_planned)
        results/classical_val_scores.csv           every hyperparameter tried and its val score
        results/analysis/lr_top_words.csv          words that push Logistic Regression to each class
        results/analysis/lr_errors.csv             misclassified test issues (most confident first)
        results/cross_repo.csv                     only with --cross-repo

Models (all baselines):
    majority                  always predicts the most common class in train ("no learning")
    naive_bayes               Multinomial Naive Bayes on word TF-IDF 1-2-grams (mandatory)
    logreg_word               Logistic Regression on word TF-IDF 1-2-grams
    logreg_word_char_stats    Logistic Regression on word + character 3-5-grams + statistical features
    linear_svm                Linear SVM (LinearSVC) on word + character + statistical features

Rules that keep the evaluation honest:
    - every vectorizer and scaler is fitted on train only (it lives inside the sklearn Pipeline)
    - hyperparameters (alpha, C) are chosen by macro-F1 on val only
    - test is predicted once, with the chosen setting, at the very end
    - random_state = 42 everywhere, so a rerun gives the same files
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cleaned"
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

RANDOM_STATE = 42

# Hyperparameter grids, searched on val only.
NB_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]
LR_CS = [0.1, 0.3, 1.0, 3.0, 10.0]
SVM_CS = [0.01, 0.03, 0.1]  # val score is flat above 0.1 and fitting gets much slower

# Keep the placeholder tokens made by clean.py ([CODE], [URL], ...) as single tokens,
# otherwise "[CODE]" would be split into the ordinary word "code".
TOKEN_PATTERN = r"\[[a-z]+\]|\b\w\w+\b"

PLACEHOLDERS = ["[CODE]", "[URL]", "[IMAGE]", "[TRACE]", "[DETAILS]"]
VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)*\b")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def statistical_features(texts):
    """Hand-made features computed from the cleaned text only (no metadata).
    All are counts with long tails (a few issues have 50+ links), so log(1 + x) is
    applied before standard scaling; otherwise a handful of outliers dominate."""
    rows = []
    for text in texts:
        title = text.split("\n", 1)[0]
        rows.append([
            len(text),                          # overall length
            len(text.split()),                  # number of words
            len(title),                         # title length
            *[text.count(tok) for tok in PLACEHOLDERS],  # code / links / images / traces / details
            text.count("?"),                    # questions ("how do I...?" often not_planned)
            text.count("!"),
            len(VERSION.findall(text)),         # version numbers (bug reports cite versions)
            text.count("\n#"),                  # template headings that were filled in
        ])
    return np.log1p(np.asarray(rows, dtype=float))


def word_tfidf():
    # 1-2-grams, words seen in at least 2 issues, sublinear tf so one long log
    # repeating a word 100 times does not dominate.
    return TfidfVectorizer(
        lowercase=True, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
        min_df=2, max_df=0.9, sublinear_tf=True,
    )


def char_tfidf():
    # Character 3-5-grams inside word boundaries: robust to typos, plural/singular,
    # identifiers such as "TypeError" or "kubectl".
    return TfidfVectorizer(
        lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
        min_df=3, max_features=300_000, sublinear_tf=True,
    )


def all_features():
    return FeatureUnion([
        ("word", word_tfidf()),
        ("char", char_tfidf()),
        ("stats", Pipeline([
            ("extract", FunctionTransformer(statistical_features)),
            ("scale", StandardScaler()),
        ])),
    ])


# ---------------------------------------------------------------------------
# Models: (features, classifier) pairs. The features do not depend on the
# hyperparameter, so they are fitted on train once and reused for the whole grid.
# ---------------------------------------------------------------------------

MODELS = [
    # name, features (unfitted), classifier for one hyperparameter value, grid, hyperparameter name
    ("majority", FunctionTransformer,  # identity: the majority class ignores the text
     lambda _: DummyClassifier(strategy="most_frequent"), [None], "-"),
    # Naive Bayes needs non-negative features, so it uses word TF-IDF only.
    ("naive_bayes", word_tfidf,
     lambda a: MultinomialNB(alpha=a), NB_ALPHAS, "alpha"),
    ("logreg_word", word_tfidf,
     lambda c: LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE), LR_CS, "C"),
    ("logreg_word_char_stats", all_features,
     lambda c: LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE), LR_CS, "C"),
    ("linear_svm", all_features,
     lambda c: LinearSVC(C=c, max_iter=10000, random_state=RANDOM_STATE), SVM_CS, "C"),
]


# ---------------------------------------------------------------------------
# Training, tuning and prediction
# ---------------------------------------------------------------------------

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def tune_on_val(name, make_features, make_clf, grid, param_name, train, val, log):
    """Fit the features on train only, then one classifier per grid value on train;
    score each on val and keep the best. Returns a fitted Pipeline (features + classifier)."""
    start = time.time()
    features = make_features()
    x_train = features.fit_transform(train["text"])   # vocabulary/IDF/scaler learned from train only
    x_val = features.transform(val["text"])
    print(f"  features fitted on train ({time.time() - start:.0f}s)")

    best_score, best_value, best_clf = -1.0, None, None
    for value in grid:
        start = time.time()
        clf = make_clf(value).fit(x_train, train["label"])
        score = macro_f1(val["label"], clf.predict(x_val))
        seconds = time.time() - start
        # No timings in the saved log, so a rerun produces byte-identical files.
        log.append({"model": name, "param": param_name, "value": value,
                    "val_macro_f1": round(score, 4)})
        print(f"  {name:24s} {param_name}={value!s:6s} val macro-F1 = {score:.4f}  ({seconds:.0f}s)")
        if score > best_score:  # ties keep the first (smaller, more regularised) value
            best_score, best_value, best_clf = score, value, clf
    print(f"  -> best {param_name} = {best_value}")
    return Pipeline([("features", features), ("clf", best_clf)]), best_value


def probability_not_planned(model, val, test):
    """P(not_planned) for every test issue.
    Naive Bayes and Logistic Regression give probabilities directly. A linear SVM only gives
    a score (signed distance to the separating line), so Platt scaling is used: a 1-feature
    logistic regression fitted on the VAL scores turns the score into a probability. It is
    monotonic, so it never changes the SVM's ranking (ROC-AUC) or its predictions."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(test["text"])[:, list(model.classes_).index(1)]
    platt = LogisticRegression().fit(model.decision_function(val["text"]).reshape(-1, 1), val["label"])
    return platt.predict_proba(model.decision_function(test["text"]).reshape(-1, 1))[:, 1]


def save_predictions(name, model, val, test):
    """Shared format for every role: id, label (true), pred, prob_not_planned. Test set only."""
    out = pd.DataFrame({
        "id": test["id"],
        "label": test["label"],
        "pred": model.predict(test["text"]),
        "prob_not_planned": np.round(probability_not_planned(model, val, test), 6),
    })
    out.to_csv(PRED_DIR / f"{name}.csv", index=False)
    print(f"  saved results/predictions/{name}.csv  (test macro-F1 = {macro_f1(out['label'], out['pred']):.4f})")
    return out


# ---------------------------------------------------------------------------
# Analysis for the presentation
# ---------------------------------------------------------------------------

def top_words(model, n=15):
    """Largest positive / negative Logistic Regression weights of the word-only model."""
    vocab = model.named_steps["features"].get_feature_names_out()
    weights = model.named_steps["clf"].coef_[0]  # positive = towards class 1 (not_planned)
    order = np.argsort(weights)
    rows = [{"direction": "not_planned", "rank": i + 1, "term": vocab[j], "weight": round(weights[j], 3)}
            for i, j in enumerate(order[::-1][:n])]
    rows += [{"direction": "completed", "rank": i + 1, "term": vocab[j], "weight": round(weights[j], 3)}
             for i, j in enumerate(order[:n])]
    return pd.DataFrame(rows)


def errors(preds, test, n=30):
    """Misclassified test issues, the most confidently wrong first."""
    df = test[["id", "repo", "text"]].merge(preds, on="id")
    df = df[df["label"] != df["pred"]].copy()
    df["confidence"] = (df["prob_not_planned"] - 0.5).abs()
    df["url"] = "https://github.com/" + df["id"].str.replace("#", "/issues/", regex=False)
    df["text"] = df["text"].str.slice(0, 300).str.replace(r"\s+", " ", regex=True)
    df = df.sort_values("confidence", ascending=False).head(n)
    return df[["id", "url", "repo", "label", "pred", "prob_not_planned", "text"]]


def cross_repo(train, test, c):
    """Train on 5 repositories, test on the 6th. Uses the same test rows as the
    in-repo benchmark, so the two macro-F1 values can be compared directly."""
    rows = []
    for repo in sorted(test["repo"].unique()):
        tr = train[train["repo"] != repo]
        te = test[test["repo"] == repo]
        model = Pipeline([
            ("features", all_features()),
            ("clf", LogisticRegression(C=c, max_iter=3000, random_state=RANDOM_STATE)),
        ]).fit(tr["text"], tr["label"])
        score = macro_f1(te["label"], model.predict(te["text"]))
        rows.append({"held_out_repo": repo, "train_rows": len(tr), "test_rows": len(te),
                     "cross_repo_macro_f1": round(score, 4)})
        print(f"  held out {repo:24s} macro-F1 = {score:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cross-repo", action="store_true", help="also run the leave-one-repository-out test")
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    log, best = [], {}
    for name, make_features, make_clf, grid, param_name in MODELS:
        print(f"\n[{name}]")
        model, value = tune_on_val(name, make_features, make_clf, grid, param_name, train, val, log)
        preds = save_predictions(name, model, val, test)
        best[name] = (model, value, preds)

    pd.DataFrame(log).to_csv(RESULTS_DIR / "classical_val_scores.csv", index=False)
    print("\nsaved results/classical_val_scores.csv")

    top_words(best["logreg_word"][0]).to_csv(ANALYSIS_DIR / "lr_top_words.csv", index=False)
    errors(best["logreg_word_char_stats"][2], test).to_csv(ANALYSIS_DIR / "lr_errors.csv", index=False)
    print("saved results/analysis/lr_top_words.csv and lr_errors.csv")

    if args.cross_repo:
        print("\n[cross-repository test: logreg_word_char_stats]")
        cross_repo(train, test, best["logreg_word_char_stats"][1]).to_csv(
            RESULTS_DIR / "cross_repo.csv", index=False)
        print("saved results/cross_repo.csv")

    print("\nDone. Next: py benchmark/run_benchmark.py")


if __name__ == "__main__":
    main()
