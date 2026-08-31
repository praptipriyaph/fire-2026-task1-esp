"""Compare two submission JSONL files: label agreement, sentence agreement,
overall summary. Useful for judging whether the retrained model diverges
suspiciously from the 420-trained baseline.
"""

import json
import sys
from pathlib import Path


def load(p: Path) -> dict:
    return {r["id"]: r for r in (json.loads(l) for l in p.open())}


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python compare_submissions.py <file_a> <file_b>", file=sys.stderr)
        return 2
    a = load(Path(sys.argv[1]))
    b = load(Path(sys.argv[2]))
    ids = sorted(set(a) & set(b))
    print(f"A: {sys.argv[1]}  ({len(a)} rows)")
    print(f"B: {sys.argv[2]}  ({len(b)} rows)")
    print(f"shared ids: {len(ids)}")

    # section-set agreement
    exact_match = 0
    jaccard_sum = 0.0
    a_only, b_only = [], []
    both_labels = []
    for i in ids:
        la = {lbl for v in a[i]["explanation"].values() for lbl in v}
        lb = {lbl for v in b[i]["explanation"].values() for lbl in v}
        if la == lb:
            exact_match += 1
        both_labels.append((i, la, lb))
        if la and lb:
            j = len(la & lb) / len(la | lb)
            jaccard_sum += j
        elif not la and not lb:
            jaccard_sum += 1.0
        a_only.extend([(i, x) for x in (la - lb)])
        b_only.extend([(i, x) for x in (lb - la)])
    print(f"\nlabel-set exact match: {exact_match}/{len(ids)} = {exact_match/len(ids):.1%}")
    print(f"label-set mean Jaccard: {jaccard_sum/len(ids):.3f}")
    print(f"labels only in A: {len(a_only)}   only in B: {len(b_only)}")

    # per-section delta
    from collections import Counter
    ca = Counter(lbl for i in ids for v in a[i]["explanation"].values() for lbl in v)
    cb = Counter(lbl for i in ids for v in b[i]["explanation"].values() for lbl in v)
    all_secs = sorted(set(ca) | set(cb))
    print(f"\nper-section counts:")
    print(f"  {'section':<12} {'A':>5} {'B':>5} {'delta':>6}")
    for s in all_secs:
        print(f"  {s:<12} {ca.get(s,0):>5} {cb.get(s,0):>5} {cb.get(s,0)-ca.get(s,0):>+6}")

    # sentence agreement (for cases where label matched)
    same_labels_same_sent = 0
    same_labels_diff_sent = 0
    for i, la, lb in both_labels:
        if la != lb:
            continue
        # both label-sets identical; check per-section sentence match
        a_map = {lbl: sent for sent, v in a[i]["explanation"].items() for lbl in v}
        b_map = {lbl: sent for sent, v in b[i]["explanation"].items() for lbl in v}
        for lbl in la:
            if a_map.get(lbl) == b_map.get(lbl):
                same_labels_same_sent += 1
            else:
                same_labels_diff_sent += 1
    tot = same_labels_same_sent + same_labels_diff_sent
    if tot:
        print(f"\nwhen label-sets match: same sentence picked = "
              f"{same_labels_same_sent}/{tot} = {same_labels_same_sent/tot:.1%}")

    # first 10 divergences
    print(f"\nfirst 10 divergences (label-set differs):")
    n = 0
    for i, la, lb in both_labels:
        if la == lb:
            continue
        print(f"  {i}  A={sorted(la)}  B={sorted(lb)}")
        n += 1
        if n >= 10:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
