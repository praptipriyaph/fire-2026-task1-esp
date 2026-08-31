"""v4-ensemble test predictor — safety-net variant of MAX-A.

Averages sigmoid probabilities from v1 (model_bert, 420-trained) and v2
(model_bert_full, 525-trained) before applying thresholds. Sentence-level
classifier scores are also averaged.

Design choice: this variant keeps everything ELSE identical to v2 —
v1's per-class thresholds, original templates.py (not templates_v3.py),
top-1 sentence per section. That makes ensembling the ONE change vs v2,
so if v3-k2's aggressive bundle underperforms on test, v4 is a stable
floor (ensembling almost never hurts).

Expected effect: variance reduction on the section probabilities → small
Macro F1 lift (+0.005 to +0.020) with essentially zero downside.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_classifier import (
    predict_probs_chunked, section_num,
    SELECTOR_BY_SECTION, SECTION_HINTS, USE_CLASSIFIER_SELECT, POOL,
    tokenize as bm25_tokenize, MIN_SENT_LEN, SENT_BATCH, predict_probs,
)
from templates import build_trace  # v2 templates, deliberately
from sentence_split import sentences as smart_sentences

WORK = Path(__file__).resolve().parent
V1_MODEL_DIR = WORK / "model_bert"          # 420-trained
V2_MODEL_DIR = WORK / "model_bert_full"     # 525-trained
DEFAULT_TEST = Path("/Users/nomitachetia/Desktop/rp/Task 1/task_1_statute_prediction_test.jsonl")
DEFAULT_OUT = WORK / "submission_task1_v4_ensemble.jsonl"


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def find_verbatim(sent: str, fact: str) -> str:
    if sent in fact:
        return sent
    norm_sent = re.sub(r"\s+", "", sent)
    if not norm_sent:
        return sent
    chars, positions = [], []
    for i, c in enumerate(fact):
        if not c.isspace():
            chars.append(c)
            positions.append(i)
    fact_flat = "".join(chars)
    idx = fact_flat.find(norm_sent)
    if idx < 0:
        return sent
    start = positions[idx]
    end = positions[idx + len(norm_sent) - 1] + 1
    return fact[start:end]


def bm25_best_with_score(fact_text, query, k1=1.5, b=0.75):
    sents = smart_sentences(fact_text)
    if not sents:
        return fact_text.strip(), 0.0
    docs = [bm25_tokenize(s) for s in sents]
    N = len(docs)
    dl = [len(d) for d in docs]
    avgdl = sum(dl) / N if N else 0.0
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    tf = [Counter(d) for d in docs]
    q_toks = bm25_tokenize(query)

    def score(i):
        if avgdl == 0:
            return 0.0
        s = 0.0
        norm = 1 - b + b * dl[i] / avgdl
        for t in q_toks:
            f = tf[i].get(t, 0)
            if f == 0:
                continue
            s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * norm)
        return s

    scores = [score(i) for i in range(N)]
    best_i = max(range(N), key=lambda i: scores[i])
    return sents[best_i], scores[best_i]


def score_all_sentences_ensemble(models, toks, test_rows, device):
    """Return {id: (sentences, ensemble_probs[n_sents, n_labels])}."""
    # collect sentences once
    per_doc_sents = {}
    flat_sents = []
    offsets = []
    for r in test_rows:
        sents = smart_sentences(r["fact"])
        kept = [s for s in sents if len(s.split()) >= MIN_SENT_LEN]
        if not kept:
            kept = sents or [r["fact"].strip()]
        start = len(flat_sents)
        flat_sents.extend(kept)
        end = len(flat_sents)
        per_doc_sents[r["id"]] = kept
        offsets.append((r["id"], start, end))

    # score with each model, average
    stacked = []
    for m, t in zip(models, toks):
        p = predict_probs(m, t, flat_sents, device,
                          batch_size=SENT_BATCH, max_len=128)
        stacked.append(p)
    ens = np.mean(np.stack(stacked, axis=0), axis=0)

    out = {}
    for did, s, e in offsets:
        out[did] = (per_doc_sents[did], ens[s:e])
    return out


def build_case(pvec, labels, thresholds, row, sent_scores):
    """Single-fact reasoning + explanation, v2-style."""
    order = list(np.argsort(-pvec))
    sel = [i for i in order if pvec[i] >= thresholds[i]]
    if not sel:
        sel = [int(order[0])]

    fact = row["fact"]
    ordered = []
    for i in sel:
        internal = labels[i]
        num = section_num(internal)
        mode = SELECTOR_BY_SECTION.get(num, "classifier") if USE_CLASSIFIER_SELECT else "bm25"
        picked = None
        if mode == "bm25":
            hint = SECTION_HINTS.get(num, "")
            bm25_pick, bm25_score = bm25_best_with_score(fact, hint)
            if bm25_score > 0:
                picked = bm25_pick
        if picked is None and sent_scores is not None:
            sents, sprobs = sent_scores[row["id"]]
            best_idx = int(np.argmax(sprobs[:, i]))
            picked = sents[best_idx]
        if picked is None:
            picked = smart_sentences(fact)[0] if smart_sentences(fact) else fact
        exact = find_verbatim(picked, fact)
        ordered.append((num, exact))

    exp = {}
    for num, sent in ordered:
        lbl = f"IPC {num}"
        if sent in exp:
            if lbl not in exp[sent]:
                exp[sent].append(lbl)
        else:
            exp[sent] = [lbl]
    reasoning = "\n\n".join(build_trace(num, sent) for num, sent in ordered)
    return exp, reasoning


def verify_output(rows, sub_rows):
    id2fact = {r["id"]: r["fact"] for r in rows}
    problems = []
    ipc_re = re.compile(r"^IPC \S+$")
    seen = set()
    for s in sub_rows:
        seen.add(s["id"])
        fact = id2fact.get(s["id"])
        if fact is None:
            problems.append(f"{s['id']}: id not in test")
            continue
        if not isinstance(s.get("reasoning_traces"), str) or not s["reasoning_traces"].strip():
            problems.append(f"{s['id']}: empty reasoning")
        exp = s.get("explanation") or {}
        if not exp:
            problems.append(f"{s['id']}: empty explanation")
        for sent, sections in exp.items():
            if sent not in fact:
                problems.append(f"{s['id']}: key not verbatim ({sent[:60]!r})")
            for lbl in sections:
                if not ipc_re.match(lbl):
                    problems.append(f"{s['id']}: bad label {lbl!r}")
    for r in rows:
        if r["id"] not in seen:
            problems.append(f"{r['id']}: missing")
    return problems


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Use v1's per-class thresholds (same as v2 inherited — safety design)
    v1_meta = json.loads((V1_MODEL_DIR / "meta.json").read_text())
    labels = v1_meta["labels"]
    per_class = v1_meta["thresholds_per_class"]
    thresholds = np.array([per_class[l] for l in labels], dtype=np.float32)
    print("v4-ensemble: v1+v2 sigmoid averaging, v1 thresholds, original templates, top-1")
    print(f"thresholds: {dict((l.split()[1], round(per_class[l], 2)) for l in labels)}")

    # Load both models
    print(f"loading v1: {V1_MODEL_DIR.name}")
    tok1 = AutoTokenizer.from_pretrained(V1_MODEL_DIR)
    m1 = AutoModelForSequenceClassification.from_pretrained(V1_MODEL_DIR).to(device)
    m1.eval()
    print(f"loading v2: {V2_MODEL_DIR.name}")
    tok2 = AutoTokenizer.from_pretrained(V2_MODEL_DIR)
    m2 = AutoModelForSequenceClassification.from_pretrained(V2_MODEL_DIR).to(device)
    m2.eval()

    test_rows = load_jsonl(DEFAULT_TEST)
    print(f"test: {len(test_rows)} cases  device: {device}")

    facts = [r["fact"] for r in test_rows]
    print("chunked case-level inference (v1)...")
    probs1, _ = predict_probs_chunked(m1, tok1, facts, device, pool=POOL)
    print("chunked case-level inference (v2)...")
    probs2, _ = predict_probs_chunked(m2, tok2, facts, device, pool=POOL)
    probs = (probs1 + probs2) / 2.0
    print(f"  ensemble mean-agreement (per-class): "
          f"{[round(float(np.mean(np.abs(probs1[:,c]-probs2[:,c]))), 3) for c in range(len(labels))]}")

    print("per-sentence ensemble scoring...")
    sent_scores = score_all_sentences_ensemble([m1, m2], [tok1, tok2], test_rows, device)

    sub_rows = []
    section_counts = Counter()
    for r, pvec in zip(test_rows, probs):
        exp, reasoning = build_case(pvec, labels, thresholds, r, sent_scores)
        sub_rows.append({
            "id": r["id"], "fact": r["fact"],
            "reasoning_traces": reasoning, "explanation": exp,
        })
        for lbls in exp.values():
            for l in lbls:
                section_counts[l] += 1

    problems = verify_output(test_rows, sub_rows)
    print(f"\nsections predicted:")
    for l, c in section_counts.most_common():
        print(f"  {l}: {c}")
    per_case_n = [len(s["explanation"]) for s in sub_rows]
    print(f"\nexplanation entries per case: mean {sum(per_case_n)/len(per_case_n):.2f} "
          f"(min={min(per_case_n)}, max={max(per_case_n)})")

    if problems:
        print(f"\nWARNING: {len(problems)} problems")
        for p in problems[:10]:
            print(f"  {p}")
    else:
        print(f"\nOK: all {len(sub_rows)} rows validate")

    with out_path.open("w") as f:
        for r in sub_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
