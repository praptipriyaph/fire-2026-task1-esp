"""Two cheap baselines that produce full-shape submissions.

Baseline 1 (`majority`): always predict the most common section in train.
Baseline 2 (`bm25`):     BM25 nearest neighbor on train facts, copy its sections.

Both write `preds_*.jsonl` in submission schema. No external dependencies
(BM25 is a ~30-line Okapi implementation).

For each predicted section we still need an `exact_fact` and `reasoning_trace`:
  - exact_fact: a verbatim sentence from the dev case's `fact`. If we have a
    reference sentence from a neighbor case (BM25 baseline), pick the dev
    sentence with the highest Jaccard token overlap to it; otherwise first sentence.
  - reasoning_trace: built from templates.build_trace(section_num, exact_fact).

Usage:
    python baselines.py            # run both, write preds, eval each
"""

import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from templates import build_trace

WORK = Path(__file__).resolve().parent
TRAIN = WORK / "train.jsonl"
DEV = WORK / "dev.jsonl"
GOLD_SUB = WORK / "dev_gold_submission.jsonl"

TOKEN_RE = re.compile(r"[a-z]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
SEC_NUM_RE = re.compile(r"Section\s+(\S+)\s+IPC")


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def write_jsonl(p: Path, rows: list) -> None:
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenize(s: str) -> list:
    return TOKEN_RE.findall(s.lower())


def sentences(text: str) -> list:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def section_num(label: str):
    m = SEC_NUM_RE.match(label)
    return m.group(1) if m else None


def pick_exact_fact(fact_text: str, ref: str = "") -> str:
    """Pick a verbatim sentence from `fact_text`. If `ref` is given, the dev
    sentence with maximum Jaccard overlap to `ref` wins; else first sentence."""
    sents = sentences(fact_text)
    if not sents:
        return fact_text.strip()
    if not ref:
        return sents[0]
    ref_toks = set(tokenize(ref))
    if not ref_toks:
        return sents[0]

    def score(s):
        st = set(tokenize(s))
        if not st:
            return 0.0
        return len(st & ref_toks) / len(st | ref_toks)

    return max(sents, key=score)


# ----- Baseline 1: majority -----

def predict_majority(dev_rows: list, train_rows: list) -> list:
    c = Counter()
    for r in train_rows:
        for s in {st["section"] for st in r["statute"]}:
            c[s] += 1
    top = c.most_common(1)[0][0]
    num = section_num(top)
    preds = []
    for r in dev_rows:
        ef = pick_exact_fact(r["fact"])
        preds.append({
            "doc_id": r["doc_id"],
            "statute": [{
                "section": top,
                "exact_fact": ef,
                "reasoning_trace": build_trace(num, ef),
            }],
        })
    return preds


# ----- Baseline 2: BM25 nearest neighbor -----

class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        df = Counter()
        for d in docs:
            for t in set(d):
                df[t] += 1
        self.idf = {
            t: math.log(1 + (self.N - df[t] + 0.5) / (df[t] + 0.5))
            for t in df
        }
        self.tf = [Counter(d) for d in docs]

    def score(self, query: list, i: int) -> float:
        if self.avgdl == 0:
            return 0.0
        s = 0.0
        norm = 1 - self.b + self.b * self.dl[i] / self.avgdl
        for t in query:
            f = self.tf[i].get(t, 0)
            if f == 0:
                continue
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + self.k1 * norm)
        return s

    def top1(self, query: list) -> int:
        return max(range(self.N), key=lambda i: self.score(query, i))


def predict_bm25(dev_rows: list, train_rows: list) -> list:
    train_tokens = [tokenize(r["fact"]) for r in train_rows]
    bm = BM25(train_tokens)
    preds = []
    for r in dev_rows:
        q = tokenize(r["fact"])
        nbr = train_rows[bm.top1(q)]
        # group neighbor's statute entries by section, keep first exact_fact as ref
        ref_by_section = {}
        for st in nbr["statute"]:
            ref_by_section.setdefault(st["section"], st["exact_fact"])
        statute = []
        for sec, ref in ref_by_section.items():
            num = section_num(sec)
            ef = pick_exact_fact(r["fact"], ref)
            statute.append({
                "section": sec,
                "exact_fact": ef,
                "reasoning_trace": build_trace(num, ef),
            })
        preds.append({"doc_id": r["doc_id"], "statute": statute})
    return preds


def run_eval(pred_path: Path) -> None:
    subprocess.run(
        ["python3", str(WORK / "eval.py"), str(GOLD_SUB), str(pred_path)],
        check=True,
    )


def main() -> int:
    train = load_jsonl(TRAIN)
    dev = load_jsonl(DEV)

    print("=" * 60)
    print("Baseline 1: majority class (always predict Section 302 IPC)")
    print("=" * 60)
    p1 = predict_majority(dev, train)
    out1 = WORK / "preds_majority.jsonl"
    write_jsonl(out1, p1)
    run_eval(out1)

    print()
    print("=" * 60)
    print("Baseline 2: BM25 nearest neighbor over train facts")
    print("=" * 60)
    p2 = predict_bm25(dev, train)
    out2 = WORK / "preds_bm25.jsonl"
    write_jsonl(out2, p2)
    run_eval(out2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
