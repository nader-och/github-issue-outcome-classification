"""
Fine-tune a pretrained transformer (Hugging Face) on the issue-outcome task.
Run on a GPU (Google Colab, T4). From the repository root:

    python training/transformers_finetune.py --model distilbert-base-uncased --name distilbert
    python training/transformers_finetune.py --model microsoft/deberta-v3-base --name deberta_v3

Quick check that everything works (1-2 minutes, saves nothing):
    python training/transformers_finetune.py --model distilbert-base-uncased --name distilbert --limit 200

Input : data/cleaned/train.csv, val.csv, test.csv  (made by preprocessing/clean.py)
Output: results/predictions/<name>.csv         test-set predictions in the shared format
                                                (id, label, pred, prob_not_planned)
        results/transformers/<name>.json        settings used + val macro-F1 after every epoch

How it stays honest (same rules as training/classical.py):
    - the model is trained on train only
    - after every epoch it is scored on val (macro-F1); the best epoch is kept
    - test is predicted once, with that best epoch, at the very end
    - seed 42 everywhere. GPU training is not perfectly deterministic, so a rerun can
      differ slightly in the last decimals; the saved predictions are the reference.

Both models use exactly the same settings (max length, learning rate, epochs, batch size),
so the comparison between the baseline (DistilBERT) and the new method (DeBERTa-v3) is fair.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cleaned"
PRED_DIR = ROOT / "results" / "predictions"
LOG_DIR = ROOT / "results" / "transformers"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face checkpoint, e.g. distilbert-base-uncased")
    p.add_argument("--name", required=True, help="short name used for the output files, e.g. distilbert")
    p.add_argument("--max-length", type=int, default=256,
                   help="tokens kept per issue (title + start of body); longer issues are cut")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="use only the first N rows of each split, for a quick test; nothing is saved")
    p.add_argument("--no-fp16", action="store_true",
                   help="disable mixed precision (use only if training gives NaN losses)")
    p.add_argument("--work-dir", default="/tmp/hf_checkpoints",
                   help="temporary checkpoints (not part of the repository)")
    return p.parse_args()


class IssueDataset(torch.utils.data.Dataset):
    """Tokenised issues + labels, in the form the Hugging Face Trainer expects."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def load_tokenizer(checkpoint):
    return AutoTokenizer.from_pretrained(checkpoint)


def load_model(checkpoint):
    # A new 2-class classification head is placed on top of the pretrained encoder.
    # .float(): some checkpoints (e.g. DeBERTa-v3) are stored in half precision (FP16).
    # Mixed-precision training needs the weights themselves in full precision (FP32),
    # otherwise it fails with "Attempting to unscale FP16 gradients".
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2,
        id2label={0: "completed", 1: "not_planned"},
        label2id={"completed": 0, "not_planned": 1},
    ).float()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def main():
    args = parse_args()
    set_seed(SEED)

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if args.limit:
        train, val, test = train.head(args.limit), val.head(args.limit), test.head(args.limit)
        print(f"QUICK TEST on {args.limit} rows per split; no files will be saved")
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    use_gpu = torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0) if use_gpu else "none (this will be very slow)")

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)
    train_ds = IssueDataset(train["text"], train["label"], tokenizer, args.max_length)
    val_ds = IssueDataset(val["text"], val["label"], tokenizer, args.max_length)
    test_ds = IssueDataset(test["text"], test["label"], tokenizer, args.max_length)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    training_args = TrainingArguments(
        output_dir=args.work_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_steps=int(0.1 * steps_per_epoch * args.epochs),  # first 10% of steps: learning rate ramps up
        eval_strategy="epoch",          # score on val after every epoch
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,    # keep the epoch with the best val macro-F1
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_gpu and not args.no_fp16,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),  # pads each batch to its longest issue
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Val macro-F1 after each epoch, from the training history.
    val_by_epoch = [round(h["eval_macro_f1"], 4) for h in trainer.state.log_history if "eval_macro_f1" in h]
    best_epoch = int(np.argmax(val_by_epoch)) + 1
    print(f"\nval macro-F1 per epoch: {val_by_epoch} -> best epoch {best_epoch}")

    # The single test prediction, with the best epoch.
    logits = trainer.predict(test_ds).predictions
    prob = softmax(logits)[:, 1]
    pred = logits.argmax(axis=1)
    test_f1 = f1_score(test["label"], pred, average="macro")
    print(f"test macro-F1 = {test_f1:.4f}")

    if args.limit:
        print("Quick test finished. Run again without --limit for the real result.")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id": test["id"],
        "label": test["label"],
        "pred": pred,
        "prob_not_planned": np.round(prob, 6),
    }).to_csv(PRED_DIR / f"{args.name}.csv", index=False)
    log = {
        "model": args.model, "name": args.name, "max_length": args.max_length,
        "epochs": args.epochs, "learning_rate": args.lr, "batch_size": args.batch_size,
        "fp16": training_args.fp16, "seed": SEED,
        "val_macro_f1_per_epoch": val_by_epoch, "best_epoch": best_epoch,
        "gpu": torch.cuda.get_device_name(0) if use_gpu else "none",
    }
    (LOG_DIR / f"{args.name}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"saved results/predictions/{args.name}.csv and results/transformers/{args.name}.json")


if __name__ == "__main__":
    main()
