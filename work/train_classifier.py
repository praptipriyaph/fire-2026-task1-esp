"""Fine-tune InLegalBERT as a 7-class multi-label classifier on the train split.

Manual training loop (transformers Trainer felt like overkill for 420 examples).
Evaluates dev each epoch with a threshold sweep (0.20-0.70) to find the best
operating point on Macro F1. Saves the best checkpoint + label list + threshold.

Output: work/model_bert/  (model, tokenizer, meta.json)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORK = Path(__file__).resolve().parent
MODEL_DIR = WORK / "model_bert"
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


def macro_f1(preds: np.ndarray, ys: np.ndarray, thresh: float) -> tuple:
    pred_bin = (preds >= thresh).astype(int)
    f1s = []
    for c in range(ys.shape[1]):
        tp = int(((pred_bin[:, c] == 1) & (ys[:, c] == 1)).sum())
        fp = int(((pred_bin[:, c] == 1) & (ys[:, c] == 0)).sum())
        fn = int(((pred_bin[:, c] == 0) & (ys[:, c] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    p_all, y_all = [], []
    for b in loader:
        out = model(
            input_ids=b["input_ids"].to(device),
            attention_mask=b["attention_mask"].to(device),
        )
        p_all.append(torch.sigmoid(out.logits).cpu().numpy())
        y_all.append(b["labels"].numpy())
    preds = np.concatenate(p_all)
    ys = np.concatenate(y_all)

    best = (-1.0, 0.5)
    for t in np.arange(0.20, 0.71, 0.05):
        m, _ = macro_f1(preds, ys, float(t))
        if m > best[0]:
            best = (m, float(t))
    m05, per05 = macro_f1(preds, ys, 0.5)
    return preds, ys, best, m05, per05


def main() -> int:
    set_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    train_rows = load_jsonl(WORK / "train.jsonl")
    dev_rows = load_jsonl(WORK / "dev.jsonl")
    print(f"train: {len(train_rows)}  dev: {len(dev_rows)}")

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    train_ds = FactDataset(train_rows, tok)
    dev_ds = FactDataset(dev_rows, tok)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_score = -1.0
    best_thresh = 0.5

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
        train_loss = tot / n

        _, _, (best_t_f1, best_t), m05, per05 = evaluate(model, dev_dl, device)
        print(
            f"epoch {epoch}  train_loss={train_loss:.4f}  "
            f"dev_F1@0.5={m05:.4f}  best_F1={best_t_f1:.4f}@t={best_t:.2f}"
        )

        if best_t_f1 > best_score:
            best_score = best_t_f1
            best_thresh = best_t
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tok.save_pretrained(MODEL_DIR)
            (MODEL_DIR / "meta.json").write_text(json.dumps({
                "labels": LABELS,
                "threshold": best_thresh,
                "best_dev_macro_f1": best_score,
                "epoch": epoch,
            }, indent=2))
            print("  -> saved (best so far)")

    print(f"\nbest dev Macro F1: {best_score:.4f} @ threshold {best_thresh:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
