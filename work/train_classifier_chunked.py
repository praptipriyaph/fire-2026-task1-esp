"""Fine-tune InLegalBERT on sliding-window CHUNKS of each case.

Same architecture and hyperparameters as train_classifier.py, but each
training case is expanded into its sliding-window chunks (stride=384,
max_len=512). Each chunk inherits the parent case's multi-hot label set,
so the model learns to predict labels from *any* position in the fact
(head, middle, or tail).

Dev evaluation uses the same chunked + mean-pool inference as the predictor,
so checkpoints are selected by the metric we'll actually deploy on.

Output: work/model_bert/  (overwrites; back up first if you want to keep
        the truncated-trained model).
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_classifier import predict_probs_chunked, POOL

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
STRIDE = 384            # 128-token overlap, matches inference
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


class ChunkedFactDataset(Dataset):
    """Expand each case into its sliding-window chunks; each chunk inherits labels."""

    def __init__(self, rows, tokenizer, max_len=MAX_LEN, stride=STRIDE):
        self.examples = []
        chunk_size = max_len - 2
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        for r in rows:
            ids = tokenizer(r["fact"], add_special_tokens=False,
                            truncation=False)["input_ids"]
            if not ids:
                ids = [tokenizer.unk_token_id]
            if len(ids) <= chunk_size:
                starts = [0]
            else:
                starts = list(range(0, len(ids) - chunk_size + 1, stride))
                if starts[-1] + chunk_size < len(ids):
                    starts.append(len(ids) - chunk_size)
            labels = {st["section"] for st in r["statute"]}
            lvec = torch.tensor(multihot(labels), dtype=torch.float32)

            for s in starts:
                chunk = ids[s:s + chunk_size]
                seq = [cls_id] + chunk + [sep_id]
                attn = [1] * len(seq)
                pad_n = max_len - len(seq)
                if pad_n > 0:
                    seq = seq + [pad_id] * pad_n
                    attn = attn + [0] * pad_n
                self.examples.append({
                    "input_ids": torch.tensor(seq, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": lvec,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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


def per_class_tuned_macro_f1(probs: np.ndarray, ys: np.ndarray):
    """Sweep per-class thresholds in [0.10, 0.70] and return tuned macro F1."""
    sweep = np.arange(0.10, 0.71, 0.05)
    best_t = np.zeros(probs.shape[1])
    for c in range(probs.shape[1]):
        bf, bt = -1.0, 0.5
        for t in sweep:
            tt = np.full(probs.shape[1], 0.5)
            tt[c] = float(t)
            pred = (probs >= tt[None, :]).astype(int)
            tp = int(((pred[:, c] == 1) & (ys[:, c] == 1)).sum())
            fp = int(((pred[:, c] == 1) & (ys[:, c] == 0)).sum())
            fn = int(((pred[:, c] == 0) & (ys[:, c] == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if f1 > bf:
                bf, bt = f1, float(t)
        best_t[c] = bt
    pred = (probs >= best_t[None, :]).astype(int)
    f1s = []
    for c in range(probs.shape[1]):
        tp = int(((pred[:, c] == 1) & (ys[:, c] == 1)).sum())
        fp = int(((pred[:, c] == 1) & (ys[:, c] == 0)).sum())
        fn = int(((pred[:, c] == 0) & (ys[:, c] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), best_t.tolist(), f1s


@torch.no_grad()
def evaluate_chunked(model, tok, dev_rows, device):
    model.eval()
    facts = [r["fact"] for r in dev_rows]
    probs, _ = predict_probs_chunked(model, tok, facts, device, pool=POOL)
    ys = np.stack([
        np.array(multihot({s["section"] for s in r["statute"]}), dtype=np.float32)
        for r in dev_rows
    ])
    macro_05, _ = macro_f1(probs, ys, 0.5)
    tuned_macro, tuned_t, _ = per_class_tuned_macro_f1(probs, ys)
    return macro_05, tuned_macro, tuned_t


def main() -> int:
    set_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  pool: {POOL}  stride: {STRIDE}")

    train_rows = load_jsonl(WORK / "train.jsonl")
    dev_rows = load_jsonl(WORK / "dev.jsonl")
    print(f"train cases: {len(train_rows)}  dev cases: {len(dev_rows)}")

    print(f"loading tokenizer + model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    print("building chunked train dataset...")
    train_ds = ChunkedFactDataset(train_rows, tok)
    print(f"train chunks: {len(train_ds)}  (avg {len(train_ds)/len(train_rows):.2f}/case)")

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_score = -1.0
    best_thresholds = None

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

        m05, tuned, tuned_t = evaluate_chunked(model, tok, dev_rows, device)
        print(f"epoch {epoch}  train_loss={train_loss:.4f}  "
              f"dev_F1@0.5={m05:.4f}  tuned_dev_F1={tuned:.4f}")

        if tuned > best_score:
            best_score = tuned
            best_thresholds = tuned_t
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tok.save_pretrained(MODEL_DIR)
            (MODEL_DIR / "meta.json").write_text(json.dumps({
                "labels": LABELS,
                "threshold": 0.5,  # legacy fallback
                "thresholds_per_class": {
                    LABELS[c]: float(tuned_t[c]) for c in range(len(LABELS))
                },
                "best_dev_macro_f1": best_score,
                "tuned_dev_macro_f1": best_score,
                "epoch": epoch,
                "trained_on": "chunked",
                "stride": STRIDE,
                "pool": POOL,
            }, indent=2))
            print("  -> saved (best so far)")

    print(f"\nbest tuned dev Macro F1: {best_score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
