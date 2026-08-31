"""Convert task1.jsonl to submission schema and produce train/dev splits.

Input  (local schema):
    {doc_id, fact, explanation: {sentence: "IPC X"}, statute: ["IPC X", ...]}

Output (submission schema, per FIRE 2026 Task 01 §06):
    {doc_id, fact, statute: [{section, exact_fact, reasoning_trace}, ...]}

Notes:
- Section labels normalized "IPC 147" -> "Section 147 IPC".
- One statute entry per (section, supporting-sentence) pair from `explanation`.
- reasoning_trace generated from per-section templates in templates.py.
- `fact` retained in train/dev files (stripped only at submission time).
- 80/20 random split, seed=42, no stratification (multi-label tiny dataset).
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

from templates import build_trace

ROOT = Path(__file__).resolve().parent.parent  # .../Task 1
RAW = ROOT / "task1.jsonl"
OUT = ROOT / "work"
DEV_FRACTION = 0.20
SEED = 42

SEC_RE = re.compile(r"IPC\s+(\S+)", re.IGNORECASE)


def normalize_section(raw: str):
    m = SEC_RE.match(raw.strip())
    if not m:
        return None, raw.strip()
    num = m.group(1)
    return num, f"Section {num} IPC"


def convert_case(rec: dict) -> dict:
    """Group explanation entries into statute list of {section, exact_fact, reasoning_trace}."""
    out = []
    for sentence, raw_sec in rec.get("explanation", {}).items():
        num, sec_norm = normalize_section(raw_sec)
        if num is None:
            continue
        out.append({
            "section": sec_norm,
            "exact_fact": sentence,
            "reasoning_trace": build_trace(num, sentence),
        })
    return {"doc_id": rec["doc_id"], "fact": rec["fact"], "statute": out}


def case_label_set(rec: dict) -> set:
    return {s["section"] for s in rec["statute"]}


def write_jsonl(path: Path, rows: list) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_distribution(rows: list) -> dict:
    c = collections.Counter()
    for r in rows:
        for sec in case_label_set(r):
            c[sec] += 1
    return dict(c.most_common())


def main() -> int:
    if not RAW.exists():
        print(f"ERROR: {RAW} not found", file=sys.stderr)
        return 1

    raw_rows = [json.loads(l) for l in RAW.open() if l.strip()]
    converted = [convert_case(r) for r in raw_rows]

    # drop cases that ended up with zero valid statute entries (none expected, but safe)
    converted = [r for r in converted if r["statute"]]
    dropped = len(raw_rows) - len(converted)

    random.seed(SEED)
    indices = list(range(len(converted)))
    random.shuffle(indices)
    dev_n = int(round(len(converted) * DEV_FRACTION))
    dev_idx = set(indices[:dev_n])

    train = [converted[i] for i in range(len(converted)) if i not in dev_idx]
    dev = [converted[i] for i in range(len(converted)) if i in dev_idx]

    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "dev.jsonl", dev)

    # also write a "gold submission" version of dev (no `fact`, mirrors test output shape)
    gold_sub = [{"doc_id": r["doc_id"], "statute": r["statute"]} for r in dev]
    write_jsonl(OUT / "dev_gold_submission.jsonl", gold_sub)

    print(f"loaded: {len(raw_rows)}  converted: {len(converted)}  dropped: {dropped}")
    print(f"split:  train={len(train)}  dev={len(dev)}  (seed={SEED}, frac={DEV_FRACTION})")
    print(f"wrote:  {OUT/'train.jsonl'}")
    print(f"        {OUT/'dev.jsonl'}")
    print(f"        {OUT/'dev_gold_submission.jsonl'}")
    print()
    print("train label dist:", label_distribution(train))
    print("dev   label dist:", label_distribution(dev))

    # show one converted example for visual check
    print("\n--- sample converted record (dev[0]) ---")
    s = dev[0]
    print(f"doc_id: {s['doc_id']}")
    print(f"fact ({len(s['fact'])} chars): {s['fact'][:160]}...")
    print(f"statute entries: {len(s['statute'])}")
    for st in s["statute"][:2]:
        print(f"  - section: {st['section']}")
        print(f"    exact_fact: {st['exact_fact'][:120]}...")
        print(f"    reasoning_trace: {st['reasoning_trace'][:160]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
