# GitHub Issue Outcome Classification

Predicting from the text of a GitHub issue whether maintainers will **resolve it** (`completed`) or **reject it** (`not_planned`): data collection, cleaning, and a benchmark of classical, neural and transformer models.

COE025 – Natural Language Processing, Project 1 (Text Classification), Istinye University.

## Task

**Input:** the title and body of a GitHub issue.
**Output:** how the issue was closed:

| Label | Meaning |
|---|---|
| `completed` | The maintainers resolved the issue (fixed or implemented). |
| `not_planned` | The maintainers rejected it (won't fix, invalid, out of scope, stale, ...). |

Predicting this early could help maintainers triage issues and help reporters write reports that get acted on.

## Dataset

### Source
Public issues from 6 large, active open-source repositories, downloaded through the official GitHub Search API by [`data/collect.py`](data/collect.py). The label comes from how the maintainers closed each issue (GitHub's `state_reason`), so no manual labelling was needed.

### Sampling
- Issues **created between January 2024 and December 2025** (24 months).
- For every repository and every month: up to **40 `completed`** and up to **40 `not_planned`** issues created in that month.
- Both classes are sampled **from the same months**, so a model cannot tell them apart by dates, version numbers or other time-related clues.

### Size

| Repository | completed | not_planned |
|---|---:|---:|
| microsoft/vscode | 960 | 960 |
| microsoft/TypeScript | 844 | 936 |
| kubernetes/kubernetes | 960 | 748 |
| flutter/flutter | 960 | 957 |
| home-assistant/core | 960 | 960 |
| numpy/numpy | 878 | 276 |
| **Total: 10,399** | **5,562 (53.5%)** | **4,837 (46.5%)** |

Overall the classes are close to balanced, but some repositories (e.g. numpy) are strongly skewed, which is one reason the benchmark uses macro-F1.

### Raw file
`data/raw/github_issues_raw.jsonl.gz` – gzip-compressed JSON Lines, one issue per line, saved **without any cleaning**. Fields:

| Field | Description |
|---|---|
| `repo`, `number`, `url` | Which issue it is |
| `title`, `body` | The issue text (the only model input) |
| `query_class` | The label: `completed` or `not_planned` |
| `state_reason` | GitHub's close reason at download time |
| `query_month`, `created_at`, `closed_at` | Dates |
| `labels`, `comments`, `author_association` | Metadata, **never used as features** (set or changed after triage, would leak the answer) |
| `user_login`, `user_type` | Author; used only to remove bot-written issues |
| `collected_at` | Download timestamp |

Load it with:
```python
import pandas as pd
df = pd.read_json("data/raw/github_issues_raw.jsonl.gz", lines=True)
```

### Reproducing the collection
```bash
pip install -r requirements.txt
export GITHUB_TOKEN=your_token          # Windows PowerShell: $env:GITHUB_TOKEN="your_token"
python data/collect.py --test           # quick check: 1 repo, 1 month, saves nothing
python data/collect.py                  # full download, about 10–15 minutes
```
GitHub data changes over time (issues get edited, reopened or deleted), so a new run may return slightly different issues. The committed raw file is the exact dataset used in this project (collected October 2026).

## Preprocessing
`preprocessing/clean.py` turns the raw file into clean, leak-free splits; every step and its counts are documented in [`preprocessing/README.md`](preprocessing/README.md).

```bash
py preprocessing/clean.py      # python3 on macOS/Linux
```

1. Keep issues whose close reason still matches the label.
2. Remove issues written by bot accounts and by two CI accounts registered as normal users (`typescript-bot`, `fluttergithubbot`).
3. Build the model input from **title + body only** and clean it: remove template comments and empty template fields; replace code blocks, links, images, collapsed `<details>` blocks and long stack traces with `[CODE]`, `[URL]`, `[IMAGE]`, `[DETAILS]` and `[TRACE]` tokens.
4. Remove near-empty issues (under 20 characters).
5. Remove duplicates before splitting: exact duplicate texts, and titles within a repository that differ only in numbers.
6. Split 80 / 10 / 10, stratified on repository + label, `random_state = 42`.

Cleaning funnel: **10,399 raw → 9,761 clean** issues (53.4% `completed`, 46.6% `not_planned`).

| Split | Rows | completed (`label = 0`) | not_planned (`label = 1`) |
|---|---:|---:|---:|
| `data/cleaned/train.csv` | 7,808 | 4,172 | 3,636 |
| `data/cleaned/val.csv` | 976 | 521 | 455 |
| `data/cleaned/test.csv` | 977 | 522 | 455 |

Columns: `id`, `repo`, `created_month`, `text`, `label`. Hyperparameters are chosen on `val`; `test` is used once, for the final benchmark.

## Models

Every model sees only the cleaned **title + body** (`text` column). All hyperparameters are chosen by macro-F1 on `val`; every setting tried is logged in [`results/classical_val_scores.csv`](results/classical_val_scores.csv).

### Classical baselines (Role 3) – `training/classical.py`

| Model | Features | Chosen setting |
|---|---|---|
| `majority` | none (always predicts the most common class) | – |
| `naive_bayes` | Multinomial Naive Bayes on word TF-IDF 1–2-grams | alpha = 0.03 |
| `logreg_word` | Logistic Regression on word TF-IDF 1–2-grams | C = 1 |
| `logreg_word_char_stats` | Logistic Regression on word 1–2-grams + character 3–5-grams + statistical features | C = 0.3 |
| `linear_svm` | Linear SVM (LinearSVC) on the same features as above | C = 0.03 |

- **Word TF-IDF:** 1–2-grams, sublinear term frequency; the `[CODE]`, `[URL]`, `[IMAGE]`, `[TRACE]` and `[DETAILS]` tokens are kept as single tokens.
- **Character TF-IDF:** 3–5-grams within word boundaries, robust to typos and code identifiers.
- **Statistical features:** text length, word count, title length, counts of each placeholder token, question marks, exclamation marks, version numbers and template headings; log-scaled, then standardised.
- Vectorizers and scalers are fitted on `train` only. The SVM's scores are turned into probabilities with Platt scaling fitted on `val`, which does not change its predictions or ranking.

### Neural and transformer models (Role 4)

_To be completed by Role 4._

## Benchmark

`benchmark/run_benchmark.py` scores every file in `results/predictions/` (one per model, test set only) in the same way and writes [`results/benchmark.md`](results/benchmark.md) and `results/benchmark.csv`.

**Primary metric: macro-F1**, the plain average of the F1 score of each class. Both classes count equally, so a model cannot score well by favouring the larger class, and the metric stays honest on skewed repositories such as numpy (76% `completed`). Accuracy, F1 per class, ROC-AUC and macro-F1 per repository are also reported. The 95% confidence interval comes from 1,000 bootstrap resamples of the test set and shows whether a gap between two models is larger than the noise.

### Results (test set, 977 issues)

| Rank | Model | Type | Macro-F1 | 95% CI | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | `naive_bayes` | baseline | **0.652** | 0.620–0.682 | 0.652 | 0.700 |
| 2 | `linear_svm` | baseline | 0.647 | 0.618–0.678 | 0.652 | 0.702 |
| 3 | `logreg_word_char_stats` | baseline | 0.646 | 0.615–0.676 | 0.651 | 0.704 |
| 4 | `logreg_word` | baseline | 0.641 | 0.610–0.672 | 0.646 | 0.714 |
| 5 | `majority` | baseline | 0.348 | 0.334–0.363 | 0.534 | – |

Per-repository scores are in [`results/benchmark.md`](results/benchmark.md).

### Analysis

- **The classical models are tied.** All four learned models score about 0.65 macro-F1, far above the majority baseline (0.35), but their confidence intervals overlap. The mandatory Naive Bayes is as strong as the rest, and the extra character and statistical features did not help.
- **What the model learns** ([`results/analysis/lr_top_words.csv`](results/analysis/lr_top_words.csv)): feature-request words such as *use case*, *proposal*, *want* and *request* push towards `not_planned`; links, screenshots and words such as *failing* and *test* push towards `completed`.
- **Cross-repository test** ([`results/cross_repo.csv`](results/cross_repo.csv)): when `logreg_word_char_stats` is trained on 5 repositories and tested on the 6th, macro-F1 drops from about 0.62 to 0.41–0.63. Part of what the model learns is repository-specific (project names such as *flutter* or *numpy* are among its strongest words) rather than a general signal of rejection.
- **Errors:** the most confidently misclassified test issues, with links, are in [`results/analysis/lr_errors.csv`](results/analysis/lr_errors.csv).

## How to run the full pipeline

Run every command from the repository root. On macOS/Linux use `python3` instead of `py`.

```
py -m pip install -r requirements.txt          # 1. install the packages
py data/collect.py                             # 2. (optional, slow) re-download the raw data; needs GITHUB_TOKEN
py preprocessing/clean.py                      # 3. clean and split -> data/cleaned/
py training/classical.py --cross-repo          # 4. classical models -> results/predictions/ (about 5 minutes)
py benchmark/run_benchmark.py                  # 5. benchmark table -> results/benchmark.md
```

Step 2 can be skipped: the raw dataset used in this project is already committed. Every script uses `random_state = 42`, so a rerun gives the same results.

## Repository structure
```
├── README.md
├── LICENSE
├── requirements.txt
├── contributions/       one file per team member
├── data/
│   ├── collect.py       downloads the raw dataset
│   ├── raw/             raw issues (unmodified)
│   └── cleaned/         train/val/test splits
├── preprocessing/       cleaning + splitting script and its documentation
├── training/            model training scripts
├── benchmark/           evaluation script
└── results/             predictions and benchmark tables
```

## Team
See the [`contributions/`](contributions/) folder.

## License
The **code** in this repository is released under the [MIT License](LICENSE).

The **issue texts** in `data/` were collected from public GitHub repositories through the official GitHub API. They remain the property of their original authors and are subject to [GitHub's Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service). They are included here only for educational and research purposes; every record links to its original issue.
