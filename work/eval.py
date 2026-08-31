"""Local eval harness for Task 1 predictions.

Computes the two metrics we can score without gold reasoning_trace:
  - Macro F1   (35% of composite, weighted heaviest)
  - Recall@3   (10%)

Both run on section labels per doc_id. Multi-label, so F1 is per-class then
macro-averaged across the 7 IPC sections in the gold.

Usage:
    python eval.py <gold.jsonl> <predictions.jsonl>

Gold and predictions must share doc_ids. Both follow the submission schema:
    {doc_id, statute: [{section, exact_fact, reasoning_trace}, ...]}

For Recall@3 we accept either:
  - the order of `statute` entries as the ranking (top = first), OR
  - a separate `top3_sections: [...]` field if present in predictions.
"""

import collections
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.open() if l.strip()]


def gold_label_sets(rows: list) -> dict:
    return {r["doc_id"]: {s["section"] for s in r["statute"]} for r in rows}


def pred_label_sets(rows: list) -> dict:
    return {r["doc_id"]: {s["section"] for s in r.get("statute", [])} for r in rows}


def pred_top3(rows: list) -> dict:
    """Top-3 ranking per doc_id. Prefer explicit `top3_sections`, else use
    order-of-appearance in `statute` (dedup, keep first)."""
    out = {}
    for r in rows:
        if "top3_sections" in r:
            out[r["doc_id"]] = list(r["top3_sections"])[:3]
            continue
        seen, ranked = set(), []
        for s in r.get("statute", []):
            sec = s["section"]
            if sec not in seen:
                seen.add(sec)
                ranked.append(sec)
            if len(ranked) == 3:
                break
        out[r["doc_id"]] = ranked
    return out


def macro_f1(gold: dict, pred: dict) -> tuple:
    """Per-class precision/recall/F1, then macro-average over classes present in gold."""
    classes = set()
    for s in gold.values():
        classes.update(s)
    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    for did, g in gold.items():
        p = pred.get(did, set())
        for c in classes:
            if c in g and c in p:
                tp[c] += 1
            elif c in p and c not in g:
                fp[c] += 1
            elif c in g and c not in p:
                fn[c] += 1
    per_class = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = (prec, rec, f1, tp[c], fp[c], fn[c])
    macro = sum(v[2] for v in per_class.values()) / len(per_class) if per_class else 0.0
    return macro, per_class


def recall_at_3(gold: dict, pred_top3_map: dict) -> float:
    """Per-case: fraction of gold sections that appear in the top-3 predictions.
    Macro-average across cases."""
    if not gold:
        return 0.0
    vals = []
    for did, g in gold.items():
        if not g:
            continue
        top3 = set(pred_top3_map.get(did, []))
        vals.append(len(g & top3) / len(g))
    return sum(vals) / len(vals) if vals else 0.0


def main(argv: list) -> int:
    if len(argv) != 3:
        print("usage: python eval.py <gold.jsonl> <predictions.jsonl>", file=sys.stderr)
        return 2

    gold_rows = load_jsonl(Path(argv[1]))
    pred_rows = load_jsonl(Path(argv[2]))

    gold = gold_label_sets(gold_rows)
    pred = pred_label_sets(pred_rows)
    p3 = pred_top3(pred_rows)

    missing = set(gold) - set(pred)
    extra = set(pred) - set(gold)
    if missing:
        print(f"WARN: {len(missing)} gold doc_ids missing from predictions")
    if extra:
        print(f"WARN: {len(extra)} predicted doc_ids not in gold (ignored)")

    macro, per_class = macro_f1(gold, pred)
    r3 = recall_at_3(gold, p3)

    print(f"n_gold:  {len(gold)}   n_pred: {len(pred)}")
    print(f"Macro F1:  {macro:.4f}")
    print(f"Recall@3:  {r3:.4f}")
    print()
    print(f"{'section':<22} {'P':>6} {'R':>6} {'F1':>6}   TP  FP  FN")
    for c in sorted(per_class):
        p, r, f1, tp, fp, fn = per_class[c]
        print(f"{c:<22} {p:6.3f} {r:6.3f} {f1:6.3f}  {tp:3d} {fp:3d} {fn:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
