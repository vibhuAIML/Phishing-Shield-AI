"""
Phishing / spam email classification on combined_phishing_email_dataset.csv
Tuned for an RTX 3050 laptop GPU (6 GB VRAM).

Examples
--------
# 1) Fast baseline (no GPU needed)
python train_phishing.py --data combined_phishing_email_dataset.csv --model tfidf

# 2) DistilBERT, random 80/10/10 split (recommended first run)
python train_phishing.py --data combined_phishing_email_dataset.csv --model distilbert-base-uncased

# 3) Leave-one-source-out: train without Enron, test on Enron
python train_phishing.py --data combined_phishing_email_dataset.csv --model distilbert-base-uncased --holdout Enron

# 4) RoBERTa-base (tight on 6 GB: small batch + gradient checkpointing)
python train_phishing.py --data combined_phishing_email_dataset.csv --model roberta-base --bs 4 --accum 8 --grad_ckpt

# Quick pipeline test on a subsample
python train_phishing.py --data combined_phishing_email_dataset.csv --model distilbert-base-uncased --subsample 10000 --epochs 1
"""
import argparse
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="distilbert-base-uncased",
                   help="'tfidf' for the baseline, or any HF model name "
                        "(distilbert-base-uncased, roberta-base, microsoft/deberta-v3-base)")
    p.add_argument("--holdout", default=None,
                   help="Source name to hold out as the test set (e.g. Enron, CEAS_08, SpamAssassin, Ling). "
                        "Omit for a random stratified split.")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=16, help="per-device batch size")
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--subsample", type=int, default=0, help="use only N rows (for quick tests)")
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves VRAM)")
    p.add_argument("--out", default="results")
    return p.parse_args()


def load_data(path, subsample):
    df = pd.read_csv(path, low_memory=False)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    # Truncate very long bodies early: only the first ~256-512 tokens are used anyway
    df["text"] = (df["subject"] + " [SEP] " + df["body"].str.slice(0, 3000)).str.strip()
    df["label"] = df["label"].astype(int)
    if subsample:
        df = df.sample(n=min(subsample, len(df)), random_state=SEED).reset_index(drop=True)
    return df[["text", "label", "source"]]


def make_splits(df, holdout):
    if holdout:
        if holdout not in set(df["source"]):
            raise SystemExit(f"Unknown source '{holdout}'. Choose from: {sorted(df['source'].unique())}")
        test = df[df["source"] == holdout]
        rest = df[df["source"] != holdout]
        train, val = train_test_split(rest, test_size=0.05, random_state=SEED, stratify=rest["label"])
    else:
        train, tmp = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
        val, test = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=tmp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def metrics_from_probs(y_true, p1):
    pred = (p1 >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, pred), "precision": prec, "recall": rec, "f1": f1}
    if len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, p1)
        out["pr_auc"] = average_precision_score(y_true, p1)
    return {k: round(float(v), 4) for k, v in out.items()}


def run_tfidf(train, val, test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(max_features=200_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=1000, C=4.0)
    clf.fit(Xtr, train["label"])
    res = {}
    for name, d in (("val", val), ("test", test)):
        p1 = clf.predict_proba(vec.transform(d["text"]))[:, 1]
        res[name] = metrics_from_probs(d["label"].values, p1)
    return res


def run_transformer(args, train, val, test):
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU will be very slow.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, df):
            enc = tok(df["text"].tolist(), truncation=True, max_length=args.max_len)
            self.ids = enc["input_ids"]
            self.mask = enc["attention_mask"]
            self.labels = df["label"].tolist()
            self.length = [len(x) for x in self.ids]  # lets group_by_length work fast

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.labels[i]}

    print("Tokenizing...")
    dtr, dva, dte = DS(train), DS(val), DS(test)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p1 = (e / e.sum(axis=1, keepdims=True))[:, 1]
        return metrics_from_probs(labels, p1)

    ta_kwargs = dict(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=torch.cuda.is_available(),
        length_column_name="length",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        dataloader_num_workers=0,  # Windows-safe
        report_to="none",
        seed=SEED,
    )
    # transformers renamed this argument in v4.41
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters \
        else "evaluation_strategy"
    ta_kwargs[key] = "epoch"

    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_params}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=dtr,
        eval_dataset=dva,
        data_collator=DataCollatorWithPadding(tok),  # dynamic padding
        compute_metrics=compute_metrics,
    )
    trainer.train()
    save_path = os.path.join(args.out, "best_distilbert")
    trainer.save_model(save_path)
    tok.save_pretrained(save_path)
    print(f"Saved best model and tokenizer to: {save_path}")
    res = {}
    for name, ds in (("val", dva), ("test", dte)):
        out = trainer.predict(ds)
        res[name] = {k.replace("test_", ""): round(float(v), 4)
                     for k, v in out.metrics.items() if k.startswith("test_") and k != "test_loss"}
    return res


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_data(args.data, args.subsample)
    train, val, test = make_splits(df, args.holdout)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"({'holdout=' + args.holdout if args.holdout else 'random split'})")

    t0 = time.time()
    res = run_tfidf(train, val, test) if args.model == "tfidf" else run_transformer(args, train, val, test)
    res["info"] = {"model": args.model, "holdout": args.holdout, "max_len": args.max_len,
                   "train_rows": len(train), "test_rows": len(test),
                   "minutes": round((time.time() - t0) / 60, 1)}

    print(json.dumps(res, indent=2))
    tag = f"{args.model.replace('/', '_')}_{args.holdout or 'random'}"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("Saved:", os.path.join(args.out, f"{tag}.json"))


if __name__ == "__main__":
    main()
