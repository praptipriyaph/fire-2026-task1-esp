"""Score exact_fact selection quality on dev.

A prediction's exact_fact is "correct" iff:
  - the predicted section matches a gold section for the same doc_id, AND
  - the predicted exact_fact sentence appears in the set of gold exact_facts
    mapped to that section in the gold dev set.

We also report a lenient match (token-Jaccard >= 0.8) for sentences that are
near-identical (whitespace / OCR drift). The strict and lenient rates bracket
the real selection quality.

Usage:
    python eval_exact_fact.py <gold.jsonl> <predictions.jsonl>
"""

import collections
import json
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z]+")


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def toks(s: str) -> set:
    return set(TOKEN_RE.findall(s.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gold_section_to_facts(rows: list) -> dict:
    """{doc_id: {section: set(exact_facts)}}"""
    out = {}
    for r in rows:
        d = collections.defaultdict(set)
        for st in r["statute"]:
            d[st["section"]].add(st["exact_fact"])
        out[r["doc_id"]] = dict(d)
    return out


def score(gold_rows: list, pred_rows: list) -> dict:
    gold = gold_section_to_facts(gold_rows)

    n_pred_entries = 0
    n_section_correct = 0
    n_strict = 0
    n_lenient = 0
    per_section = collections.defaultdict(lambda: [0, 0, 0])  # [correct_section, strict, lenient]

    for pr in pred_rows:
        did = pr["doc_id"]
        gold_secs = gold.get(did, {})
        for st in pr.get("statute", []):
            n_pred_entries += 1
            sec = st["section"]
            ef = st["exact_fact"].strip()
            if sec not in gold_secs:
                continue
            n_section_correct += 1
            per_section[sec][0] += 1

            gold_efs = gold_secs[sec]
            # strict: exact substring match either direction (handles minor whitespace)
            strict = any(ef == g or ef in g or g in ef for g in gold_efs)
            if strict:
                n_strict += 1
                per_section[sec][1] += 1

            # lenient: token-jaccard >= 0.8 with any gold ef for this section
            pt = toks(ef)
            lenient = strict or any(jaccard(pt, toks(g)) >= 0.8 for g in gold_efs)
            if lenient:
                n_lenient += 1
                per_section[sec][2] += 1

    return {
        "n_pred_entries": n_pred_entries,
        "n_section_correct": n_section_correct,
        "strict_rate": n_strict / n_section_correct if n_section_correct else 0.0,
        "lenient_rate": n_lenient / n_section_correct if n_section_correct else 0.0,
        "per_section": dict(per_section),
    }


def main(argv: list) -> int:
    if len(argv) != 3:
        print("usage: python eval_exact_fact.py <gold.jsonl> <predictions.jsonl>", file=sys.stderr)
        return 2
    gold_rows = load_jsonl(Path(argv[1]))
    pred_rows = load_jsonl(Path(argv[2]))
    r = score(gold_rows, pred_rows)

    print(f"predicted entries:           {r['n_pred_entries']}")
    print(f"of which section is correct: {r['n_section_correct']}")
    print(f"exact_fact strict  match:    {r['strict_rate']:.4f}  "
          f"({int(r['strict_rate']*r['n_section_correct'])}/{r['n_section_correct']})")
    print(f"exact_fact lenient match:    {r['lenient_rate']:.4f}  "
          f"({int(r['lenient_rate']*r['n_section_correct'])}/{r['n_section_correct']})")
    print()
    print(f"{'section':<22} {'cor_sec':>7} {'strict':>7} {'lenient':>7}")
    for sec in sorted(r["per_section"]):
        cs, st, le = r["per_section"][sec]
        s_rate = st / cs if cs else 0.0
        l_rate = le / cs if cs else 0.0
        print(f"{sec:<22} {cs:7d} {s_rate:7.3f} {l_rate:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
