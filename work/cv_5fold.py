"""5-fold cross-validation on the full 525-case pool.

Purpose: give a defensible mean ± std Macro F1 for the submitted v2
configuration, since v2 was trained on all 525 cases and therefore has
no held-out F1 to report in the working notes.

Procedure per fold:
  - Train InLegalBERT on 4 folds (~420 cases), same hyperparameters as
    train_classifier.py (10 epochs, AdamW lr=2e-5, batch=8, max_len=512).
  - Chunked mean-pool inference on the held-out fold (~105 cases).
  - Report three F1 numbers so the paper can pick which to headline:
      1. F1 @ global threshold 0.5      (deployment-independent baseline)
      2. F1 @ v2's frozen per-class     (what the submitted model actually uses)
      3. F1 @ per-fold tuned per-class  (ceiling of the pipeline on that split)
  - Also report Recall@3 (rank-based, threshold-independent).

Outputs: cv_5fold_results.json with per-fold numbers + mean/std.

Runtime: ~25 min/fold * 5 = ~2h 5m training + ~10 min eval on MPS.
Fold models are NOT saved to disk to keep the checkout light.
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
MODEL_NAME = "law-ai/InLegalBERT"
FROZEN_META = WORK / "model_bert" / "meta.json"   # v1's tuned thresholds (also used by v2)
OUT = WORK / "cv_5fold_results.json"

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

K = 5
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


def kfold_indices(n: int, k: int, seed: int) -> list:
    """Return k lists of test indices (union covers 0..n-1, disjoint)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    fold_size = n // k
    remainder = n % k
    folds, start = [], 0
    for i in range(k):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(idx[start:start + size])
        start += size
    return folds


def macro_f1(probs: np.ndarray, ys: np.ndarray, thresholds: np.ndarray) -> tuple:
    pred = (probs >= thresholds[None, :]).astype(int)
    f1s = []
    for c in range(ys.shape[1]):
        tp = int(((pred[:, c] == 1) & (ys[:, c] == 1)).sum())
        fp = int(((pred[:, c] == 1) & (ys[:, c] == 0)).sum())
        fn = int(((pred[:, c] == 0) & (ys[:, c] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s)), f1s


def tune_per_class(probs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sweep 0.10-0.70 per class independently to maximize per-class F1."""
    best_t = np.full(probs.shape[1], 0.5)
    for c in range(probs.shape[1]):
        bf, bt = -1.0, 0.5
        for t in np.arange(0.10, 0.71, 0.05):
            tt = best_t.copy()
            tt[c] = float(t)
            _, f1s = macro_f1(probs, ys, tt)
            if f1s[c] > bf:
                bf, bt = f1s[c], float(t)
        best_t[c] = bt
    return best_t


def recall_at_k(probs: np.ndarray, ys: np.ndarray, k: int = 3) -> float:
    """For each case: is EVERY gold label in the top-k predictions?
    Cases with zero gold labels are excluded from the denominator."""
    hits, total = 0, 0
    for i in range(len(probs)):
        gold = set(np.where(ys[i] == 1)[0].tolist())
        if not gold:
            continue
        topk = set(np.argsort(-probs[i])[:k].tolist())
        # Recall@k: fraction of gold labels caught in top-k, then averaged
        # (macro-style; different from "all gold in top-k")
        hits += len(gold & topk) / len(gold)
        total += 1
    return hits / total if total else 0.0


@torch.no_grad()
def evaluate(model, tok, eval_rows: list, device: str) -> tuple:
    facts = [r["fact"] for r in eval_rows]
    probs, _ = predict_probs_chunked(model, tok, facts, device, pool=POOL)
    ys = np.stack([
        np.array(multihot({s["section"] for s in r["statute"]}), dtype=np.float32)
        for r in eval_rows
    ])
    return probs, ys


def train_fold(train_rows: list, device: str) -> tuple:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    ).to(device)
    ds = FactDataset(train_rows, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl:
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
        print(f"      epoch {epoch:2d}  loss={tot/n:.4f}", flush=True)
    return model, tok


def main() -> int:
    set_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  K={K}  epochs={EPOCHS}  seed={SEED}  pool={POOL}")

    # Merge train + dev into the full 525-case pool
    train_rows = load_jsonl(WORK / "train.jsonl")
    dev_rows = load_jsonl(WORK / "dev.jsonl")
    all_rows = train_rows + dev_rows
    print(f"total cases: {len(all_rows)}  (train={len(train_rows)} + dev={len(dev_rows)})")

    # Load v2's frozen thresholds (inherited from v1 tuning)
    frozen_meta = json.loads(FROZEN_META.read_text())
    frozen_thresh = np.array(
        [frozen_meta["thresholds_per_class"][l] for l in LABELS], dtype=np.float32
    )
    print(f"frozen thresholds: {dict(zip([l.split()[1] for l in LABELS], frozen_thresh.tolist()))}")

    folds = kfold_indices(len(all_rows), K, SEED)
    fold_sizes = [len(f) for f in folds]
    print(f"fold sizes: {fold_sizes} (sum={sum(fold_sizes)})")

    results = {
        "config": {
            "K": K, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
            "max_len": MAX_LEN, "seed": SEED, "pool": POOL,
            "frozen_thresholds": {LABELS[c]: float(frozen_thresh[c]) for c in range(len(LABELS))},
        },
        "folds": [],
    }

    for fold_i in range(K):
        print(f"\n=== fold {fold_i + 1}/{K} ===", flush=True)
        set_seed(SEED + fold_i)  # different shuffle per fold, but reproducible
        eval_idx = folds[fold_i]
        train_idx = [i for j in range(K) if j != fold_i for i in folds[j]]
        tr_rows = [all_rows[i] for i in train_idx]
        ev_rows = [all_rows[i] for i in eval_idx]
        print(f"  train: {len(tr_rows)}  eval: {len(ev_rows)}", flush=True)

        model, tok = train_fold(tr_rows, device)
        probs, ys = evaluate(model, tok, ev_rows, device)

        # Three F1s
        f1_05, per_05 = macro_f1(probs, ys, np.full(len(LABELS), 0.5))
        f1_frozen, per_frozen = macro_f1(probs, ys, frozen_thresh)
        tuned_thresh = tune_per_class(probs, ys)
        f1_tuned, per_tuned = macro_f1(probs, ys, tuned_thresh)
        r_at_3 = recall_at_k(probs, ys, k=3)

        fold_result = {
            "fold": fold_i + 1,
            "n_train": len(tr_rows), "n_eval": len(ev_rows),
            "macro_f1_at_05": f1_05,
            "macro_f1_at_frozen": f1_frozen,
            "macro_f1_at_tuned": f1_tuned,
            "recall_at_3": r_at_3,
            "per_class_f1_at_frozen": {LABELS[c]: per_frozen[c] for c in range(len(LABELS))},
            "per_class_f1_at_tuned": {LABELS[c]: per_tuned[c] for c in range(len(LABELS))},
            "tuned_thresholds": {LABELS[c]: float(tuned_thresh[c]) for c in range(len(LABELS))},
        }
        results["folds"].append(fold_result)
        print(f"  F1@0.5={f1_05:.4f}  F1@frozen={f1_frozen:.4f}  "
              f"F1@tuned={f1_tuned:.4f}  R@3={r_at_3:.4f}", flush=True)

        # Free memory before next fold
        del model, tok
        if device == "mps":
            torch.mps.empty_cache()

        # Save partial results after every fold in case of crash
        OUT.write_text(json.dumps(results, indent=2))

    # Aggregate
    arr = lambda k: np.array([f[k] for f in results["folds"]])
    for key in ["macro_f1_at_05", "macro_f1_at_frozen", "macro_f1_at_tuned", "recall_at_3"]:
        v = arr(key)
        results.setdefault("summary", {})[key] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    # Per-class aggregates
    per_class_agg = {}
    for scheme in ["per_class_f1_at_frozen", "per_class_f1_at_tuned"]:
        per_class_agg[scheme] = {}
        for c, lbl in enumerate(LABELS):
            vals = np.array([f[scheme][lbl] for f in results["folds"]])
            per_class_agg[scheme][lbl] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
            }
    results["summary"]["per_class"] = per_class_agg

    OUT.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("5-FOLD CV SUMMARY")
    print("=" * 60)
    for key in ["macro_f1_at_05", "macro_f1_at_frozen", "macro_f1_at_tuned", "recall_at_3"]:
        s = results["summary"][key]
        print(f"  {key:<24}  {s['mean']:.4f} ± {s['std']:.4f}   "
              f"(min={s['min']:.4f}, max={s['max']:.4f})")
    print(f"\nfull results: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
