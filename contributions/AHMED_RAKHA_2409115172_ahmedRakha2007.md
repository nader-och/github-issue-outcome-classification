# Contribution – Ahmed Rakha

- **Student ID:** 2409115172
- **GitHub:** [ahmedRakha2007](https://github.com/ahmedRakha2007)
- **Role:** Transformer models (Role 4)

## How the team worked

The three of us (Nader Ochen, Çınar Sabancı, Ahmed Rakha) worked together in joint sessions on Nader's computer. For that reason, the commits for my part were pushed from Nader's GitHub account (`nader-och`); they are listed at the end of this file.

## Summary

I was responsible for fine-tuning the transformer models: DistilBERT as the baseline and DeBERTa-v3 as our new method, trained on a GPU in Google Colab.

## Fine-tuning script (`training/transformers_finetune.py`)

- One script fine-tunes any Hugging Face checkpoint for the 2-class task; the model is chosen with `--model`.
- **Architecture vs. specific model:** the architectures are DistilBERT and DeBERTa-v3; the specific models are `distilbert-base-uncased` (baseline, the checkpoint used in class) and `microsoft/deberta-v3-base` (new method, not covered in class).
- **Fair comparison:** both models use identical settings: first 256 tokens of title + body, learning rate 2e-5, batch size 16, 5 epochs, 10% warm-up, weight decay 0.01, mixed precision (FP16), seed 42.
- **Leak-free protocol:** after every epoch the model is scored on validation; the epoch with the best validation macro-F1 is kept and used once on the test set.
- Saves test predictions in the same format as the classical models, so the shared benchmark scores them identically, and logs the settings and per-epoch validation scores in `results/transformers/`.
- Includes a `--limit` quick-test mode to check the whole pipeline in about 2 minutes before a full run.

## Problems solved

- **FP16 crash with DeBERTa-v3:** its checkpoint is stored in half precision, and the current `transformers` version loads it that way, so mixed-precision training failed with "Attempting to unscale FP16 gradients". Fixed by converting the model to full precision (FP32) after loading; DistilBERT was unaffected.
- **Number of epochs:** with 3 epochs, DeBERTa's validation score was still rising at the last epoch, so both models were retrained with 5 epochs (the best epoch is still chosen on validation, never on test).

## Why DeBERTa-v3 as the new method

- **Disentangled attention:** each word's content and its position are encoded as separate vectors.
- **ELECTRA-style pre-training:** it learns by detecting replaced tokens instead of predicting masked ones.

## Results (test set, macro-F1)

| Model | Best epoch (val) | Val macro-F1 | Test macro-F1 |
|---|---|---:|---:|
| DistilBERT (baseline) | 3 of 5 | 0.651 | 0.649 |
| DeBERTa-v3 (new method) | 4 of 5 | 0.671 | 0.641 |

- DeBERTa-v3 had the best validation score of any model, but on test it is statistically tied with the classical models.
- Both transformers find rejected issues more evenly than logistic regression and the SVM (F1 on `not_planned` 0.62–0.64 vs 0.60–0.61).

## Documentation
- Wrote the **Neural and transformer models** part of the Models section and the Colab instructions in **How to run** of the main `README.md`.

## Commits (pushed from `nader-och` during our joint sessions)

| Commit | Description |
|---|---|
| `a66cdc7` | Add transformer fine-tuning script |
| `7ac6f1a` | Add transformer results (DistilBERT, DeBERTa-v3) and updated benchmark |
| `8b2a29c` | Update README with transformer models and final benchmark |
