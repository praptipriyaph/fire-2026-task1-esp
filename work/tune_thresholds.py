"""Tune one threshold per IPC section on dev to maximize per-class F1.

Loads the trained classifier, predicts dev probabilities once, then sweeps
0.10-0.70 (step 0.05) independently for each class. Saves the chosen
thresholds back into model_bert/meta.json under `thresholds_per_class`.

Note: tuning on dev *does* fit to dev. With only 105 dev cases and 7 classes,
the per-class thresholds are a coarse calibration, not a precision instrument.
We log old-vs-new F1 so any regression is visible.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_classifier import predict_probs_chunked, CHUNK_FACTS, POOL

WORK = Path(__file__).resolve().parent
MODEL_DIR = WORK / "model_bert"
DEV = WORK / "dev.jsonl"
MAX_LEN = 512
BATCH = 8


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def multihot(sections: set, labels: list) -> np.ndarray:
    v = np.zeros(len(labels), dtype=np.float32)
    for s in sections:
        if s in labels:
            v[labels.index(s)] = 1.0
    return v


@torch.no_grad()
def predict_probs(model, tok, texts, device):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def per_class_f1(probs: np.ndarray, ys: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """thresholds: shape [C]. Returns per-class F1, shape [C]."""
    pred = (probs >= thresholds[None, :]).astype(int)
    f1s = np.zeros(probs.shape[1])
    for c in range(probs.shape[1]):
        tp = int(((pred[:, c] == 1) & (ys[:, c] == 1)).sum())
        fp = int(((pred[:, c] == 1) & (ys[:, c] == 0)).sum())
        fn = int(((pred[:, c] == 0) & (ys[:, c] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[c] = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1s


def main() -> int:
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    labels = meta["labels"]
    old_thresh = float(meta["threshold"])
    print(f"loaded model: labels={len(labels)}  old global thresh={old_thresh:.2f}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    dev_rows = load_jsonl(DEV)
    facts = [r["fact"] for r in dev_rows]
    if CHUNK_FACTS:
        print(f"using chunked inference (pool={POOL})")
        probs, _ = predict_probs_chunked(model, tok, facts, device, pool=POOL)
    else:
        probs = predict_probs(model, tok, facts, device)
    ys = np.stack([multihot({s["section"] for s in r["statute"]}, labels)
                   for r in dev_rows])
    print(f"probs shape: {probs.shape}  ys shape: {ys.shape}")

    # baseline: global threshold
    base_thresh = np.full(len(labels), old_thresh)
    base_f1 = per_class_f1(probs, ys, base_thresh)
    base_macro = float(base_f1.mean())
    print(f"\nbaseline (global thresh={old_thresh:.2f}) macro F1 = {base_macro:.4f}")
    for c, lbl in enumerate(labels):
        print(f"  {lbl:<22}  F1={base_f1[c]:.3f}")

    # per-class sweep
    sweep = np.arange(0.10, 0.71, 0.05)
    best_thresh = np.zeros(len(labels))
    best_f1_per = np.zeros(len(labels))
    for c in range(len(labels)):
        best_t, best_f = old_thresh, -1.0
        for t in sweep:
            tt = base_thresh.copy()
            tt[c] = float(t)
            f1 = per_class_f1(probs, ys, tt)[c]
            if f1 > best_f:
                best_f = f1
                best_t = float(t)
        best_thresh[c] = best_t
        best_f1_per[c] = best_f

    tuned_f1 = per_class_f1(probs, ys, best_thresh)
    tuned_macro = float(tuned_f1.mean())

    print(f"\ntuned macro F1 = {tuned_macro:.4f}  (delta {tuned_macro - base_macro:+.4f})")
    print(f"{'section':<22}  {'old_t':>5}  {'old_F1':>6}  {'new_t':>5}  {'new_F1':>6}")
    for c, lbl in enumerate(labels):
        print(f"  {lbl:<22}  {old_thresh:5.2f}  {base_f1[c]:6.3f}  "
              f"{best_thresh[c]:5.2f}  {tuned_f1[c]:6.3f}")

    meta["thresholds_per_class"] = {labels[c]: float(best_thresh[c])
                                    for c in range(len(labels))}
    meta["tuned_dev_macro_f1"] = tuned_macro
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved per-class thresholds to {MODEL_DIR/'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
