# Contribution – Çınar Sabancı

- **Student ID:** 2409115637
- **GitHub:** [cyh77](https://github.com/cyh77)
- **Role:** Classical models and benchmark (Role 3)

## How the team worked

The three of us (Nader Ochen, Çınar Sabancı, Ahmed Rakha) worked together in joint sessions on Nader's computer. For that reason, the commits for my part were pushed from Nader's GitHub account (`nader-och`); they are listed at the end of this file.

## Summary

I was responsible for the classical baseline models and for the benchmark that scores every model in the project the same way, including the choice of evaluation metric.

## Classical models (`training/classical.py`)

- Five baselines: a **majority-class** model (the "no learning" reference), **Multinomial Naive Bayes** on word TF-IDF 1–2-grams, **Logistic Regression** on word TF-IDF, **Logistic Regression** on word + character 3–5-grams + statistical features, and a **Linear SVM** on the same combined features.
- Statistical features computed from the cleaned text only: length, word count, title length, counts of each placeholder token (`[CODE]`, `[URL]`, ...), question marks, exclamation marks, version numbers and template headings, log-scaled and standardised.
- The placeholder tokens are kept as single tokens, so `[CODE]` is not merged with the ordinary word "code".
- **Leak-free protocol:** vectorizers and scalers are fitted on train only; `alpha` and `C` are chosen by macro-F1 on validation; the test set is predicted once, at the end. Every setting tried is logged in `results/classical_val_scores.csv`.
- The SVM's scores are converted into probabilities with Platt scaling fitted on validation, which does not change its predictions or ranking.
- **Analysis outputs:** the strongest words for each class (`results/analysis/lr_top_words.csv`), the most confidently misclassified test issues with links (`results/analysis/lr_errors.csv`), and a **cross-repository test** (`results/cross_repo.csv`): train on 5 repositories, test on the 6th.
- **Reproducibility:** `random_state = 42` everywhere; a rerun produces byte-identical files.

## Benchmark (`benchmark/run_benchmark.py`)

- Scores every prediction file in `results/predictions/` with the same code, after checking that its ids and labels match `test.csv`.
- **Primary metric: macro-F1**, chosen because both classes must count equally: a majority-class model reaches 53% accuracy but only 0.35 macro-F1, and some repositories are strongly skewed (NumPy is 76% `completed`).
- Also reports accuracy, F1 per class, ROC-AUC, macro-F1 per repository, and a **95% bootstrap confidence interval** (1,000 resamples) to show whether gaps between models are larger than the noise.
- Writes `results/benchmark.csv` and `results/benchmark.md`; new models are added with one line in `MODEL_INFO`.

## Results and findings

- Classical models: Naive Bayes 0.652, Linear SVM 0.647, Logistic Regression 0.641–0.646 macro-F1, against 0.348 for the majority baseline.
- All learned models' confidence intervals overlap, so they are statistically tied.
- Cross-repository test: macro-F1 drops from about 0.62 to 0.41–0.63, showing that part of what the models learn is repository-specific.

## Documentation
- Wrote the **Classical baselines** part of the Models section and the **Benchmark** section (metric choice, results, analysis) of the main `README.md`.

## Commits (pushed from `nader-och` during our joint sessions)

| Commit | Description |
|---|---|
| `e072779` | Add classical models, benchmark results and dependencies |
| `08e326b` | Fill README: models, benchmark, how to run |
