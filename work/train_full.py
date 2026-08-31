"""Retrain the InLegalBERT classifier on the FULL 525-case pool (train + dev).

Same hyperparameters as train_classifier.py — 10 epochs, AdamW lr=2e-5,
batch 8, max_len 512, seed 42. Difference: no dev holdout (all 525 used
for training), so we save the FINAL epoch's weights (not "best" — there's
no held-out signal to select against).

Reuses per-class thresholds from the original 420-trained model_bert/
without retuning. Rationale: the tuned thresholds were fit on dev of a
model that's very similar in calibration; adding 25% more training data
usually shifts probability outputs only slightly. Retuning on test would
be data leakage, and we have no other holdout.

Output: work/model_bert_full/
"""

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORK = Path(__file__).resolve().parent
SRC_MODEL_DIR = WORK / "model_bert"          # for stealing tuned thresholds
OUT_MODEL_DIR = WORK / "model_bert_full"
MODEL_NAME = "law-ai/InLegalBERT"

LABELS = [
    "Section 147 IPC",
    "Section 201 IPC",
    "Section 302 IPC",
    "Section 376 IPC",
    "Section 420 IPC",
    "Section 498A IPC",
    "Section 506 IPC",
]
L2I = {l: i for i, l in enumerate(LABELS)}

EPOCHS = 10
BATCH = 8
LR = 2e-5
MAX_LEN = 512
SEED = 42


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def multihot(sections: set) -> list:
    v = [0.0] * len(LABELS)
    for s in sections:
        if s in L2I:
            v[L2I[s]] = 1.0
    return v


class FactDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["fact"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        labels = {st["section"] for st in r["statute"]}
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(multihot(labels), dtype=torch.float32),
        }


def main() -> int:
    set_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    train_rows = load_jsonl(WORK / "train.jsonl")
    dev_rows = load_jsonl(WORK / "dev.jsonl")
    all_rows = train_rows + dev_rows
    random.Random(SEED).shuffle(all_rows)
    print(f"train_full: {len(all_rows)} (train={len(train_rows)} + dev={len(dev_rows)})")

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    train_ds = FactDataset(all_rows, tok)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for b in train_dl:
            opt.zero_grad()
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            bs = b["labels"].size(0)
            tot += out.loss.item() * bs
            n += bs
        print(f"epoch {epoch}  train_loss={tot/n:.4f}")

    # Save final model
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_MODEL_DIR)
    tok.save_pretrained(OUT_MODEL_DIR)

    # Steal tuned thresholds from the 420-trained model
    src_meta = json.loads((SRC_MODEL_DIR / "meta.json").read_text())
    tuned = src_meta.get("thresholds_per_class")
    if not tuned:
        print("WARNING: source meta.json has no thresholds_per_class — using 0.5")
        tuned = {l: 0.5 for l in LABELS}
    (OUT_MODEL_DIR / "meta.json").write_text(json.dumps({
        "labels": LABELS,
        "threshold": src_meta.get("threshold", 0.5),
        "thresholds_per_class": tuned,
        "best_dev_macro_f1": src_meta.get("best_dev_macro_f1", 0.0),
        "tuned_dev_macro_f1": src_meta.get("tuned_dev_macro_f1", 0.0),
        "epoch": EPOCHS,
        "trained_on": "train+dev merged (525 cases, no dev holdout)",
        "thresholds_inherited_from": str(SRC_MODEL_DIR),
    }, indent=2))
    print(f"saved model + thresholds to {OUT_MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
