# Benchmark (test set, 977 issues)

Sorted by macro-F1 (primary metric). CI = 95% bootstrap confidence interval of macro-F1.

| Rank | Model | Type | Macro-F1 | 95% CI | Accuracy | F1 completed | F1 not_planned | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | `naive_bayes` | baseline | **0.652** | 0.620–0.682 | 0.652 | 0.661 | 0.643 | 0.700 |
| 2 | `linear_svm` | baseline | **0.647** | 0.618–0.678 | 0.652 | 0.688 | 0.607 | 0.702 |
| 3 | `logreg_word_char_stats` | baseline | **0.646** | 0.615–0.676 | 0.651 | 0.687 | 0.606 | 0.704 |
| 4 | `logreg_word` | baseline | **0.641** | 0.610–0.672 | 0.646 | 0.682 | 0.600 | 0.714 |
| 5 | `majority` | baseline | **0.348** | 0.334–0.363 | 0.534 | 0.696 | 0.000 | – |

## Macro-F1 per repository

| Model | flutter | core | kubernetes | TypeScript | vscode | numpy |
|---|---|---|---|---|---|---|
| `naive_bayes` | 0.604 | 0.635 | 0.721 | 0.582 | 0.597 | 0.463 |
| `linear_svm` | 0.624 | 0.619 | 0.727 | 0.587 | 0.610 | 0.463 |
| `logreg_word_char_stats` | 0.626 | 0.619 | 0.727 | 0.581 | 0.610 | 0.463 |
| `logreg_word` | 0.614 | 0.584 | 0.743 | 0.568 | 0.632 | 0.459 |
| `majority` | 0.315 | 0.333 | 0.361 | 0.326 | 0.336 | 0.433 |

numpy has only 115 test issues (27 not_planned), so its per-repository score is noisy.
