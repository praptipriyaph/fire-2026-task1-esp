"""v3 test predictor — MAX-A strategy.

Same core pipeline as predict_test.py, but with three v3 changes:
  1. Reads CV-median thresholds from meta_cvmed.json (instead of the
     v1-inherited per-class thresholds)
  2. Uses templates_v3.py (polished legal prose + section-specific vocab)
  3. Supports TOP_K_SENTENCES per section — when K=2, picks the top-2
     classifier-scored sentences per predicted section and weaves both
     into the reasoning template. When K=1, behaves like predict_test.py.

Usage:
    python predict_test_v3.py <k> [<out_path>]
where k is 1 or 2. Default out: submission_task1_v3_k{K}.jsonl
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
from templates_v3 import build_trace
from sentence_split import sentences as smart_sentences

WORK = Path(__file__).resolve().parent
MODEL_DIR = WORK / "model_bert_full"
META_FILE = MODEL_DIR / "meta_cvmed.json"   # CV-median thresholds
DEFAULT_TEST = Path("/Users/nomitachetia/Desktop/rp/Task 1/task_1_statute_prediction_test.jsonl")


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


def score_all_sentences_by_id(model, tok, test_rows, device):
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
    probs = predict_probs(model, tok, flat_sents, device,
                          batch_size=SENT_BATCH, max_len=128)
    out = {}
    for did, s, e in offsets:
        out[did] = (per_doc_sents[did], probs[s:e])
    return out


def pick_top_k_sentences(row, label_idx, sent_scores, top_k):
    """Return up to top_k UNIQUE verbatim sentences for label i, ranked by
    classifier probability for that label."""
    sents, sprobs = sent_scores[row["id"]]
    order = np.argsort(-sprobs[:, label_idx])
    picked = []
    seen = set()
    for idx in order:
        s = sents[int(idx)]
        v = find_verbatim(s, row["fact"])
        if v in seen:
            continue
        picked.append(v)
        seen.add(v)
        if len(picked) >= top_k:
            break
    return picked


def build_reasoning_and_explanation(pvec, labels, thresholds, row, sent_scores, top_k):
    """Return (explanation_dict, reasoning_traces_str)."""
    order = list(np.argsort(-pvec))
    sel = [i for i in order if pvec[i] >= thresholds[i]]
    if not sel:
        sel = [int(order[0])]

    fact = row["fact"]
    exp = {}
    reasoning_paras = []
    for i in sel:
        internal = labels[i]
        num = section_num(internal)
        mode = SELECTOR_BY_SECTION.get(num, "classifier") if USE_CLASSIFIER_SELECT else "bm25"

        # Primary sentence via hybrid rule (unchanged from v2 for stability)
        primary = None
        if mode == "bm25":
            hint = SECTION_HINTS.get(num, "")
            bm25_pick, bm25_score = bm25_best_with_score(fact, hint)
            if bm25_score > 0:
                primary = find_verbatim(bm25_pick, fact)
        if primary is None:
            # classifier top-1 for this label
            sents, sprobs = sent_scores[row["id"]]
            best_idx = int(np.argmax(sprobs[:, i]))
            primary = find_verbatim(sents[best_idx], fact)

        # Secondary sentence (only if top_k >= 2) via classifier top-2 excluding primary
        secondary = None
        if top_k >= 2:
            k_sents = pick_top_k_sentences(row, i, sent_scores, top_k=3)
            for s in k_sents:
                if s != primary:
                    secondary = s
                    break

        # Build reasoning with 1 or 2 fact sentences woven in
        facts_for_template = [primary] if secondary is None else [primary, secondary]
        reasoning_paras.append(build_trace(num, facts_for_template))

        # Populate explanation dict — primary always added; secondary added
        # only if distinct and top_k>=2
        submission_lbl = f"IPC {num}"
        for s in [primary] + ([secondary] if secondary else []):
            if s in exp:
                if submission_lbl not in exp[s]:
                    exp[s].append(submission_lbl)
            else:
                exp[s] = [submission_lbl]

    reasoning = "\n\n".join(reasoning_paras)
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
            problems.append(f"{s['id']}: id not in test set")
            continue
        if not isinstance(s.get("reasoning_traces"), str) or not s["reasoning_traces"].strip():
            problems.append(f"{s['id']}: empty reasoning_traces")
        exp = s.get("explanation") or {}
        if not exp:
            problems.append(f"{s['id']}: empty explanation")
        for sent, sections in exp.items():
            if sent not in fact:
                problems.append(f"{s['id']}: key not verbatim in fact ({sent[:60]!r})")
            for lbl in (sections or []):
                if not ipc_re.match(lbl):
                    problems.append(f"{s['id']}: bad label {lbl!r}")
    for r in rows:
        if r["id"] not in seen:
            problems.append(f"{r['id']}: missing")
    return problems


def main() -> int:
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if top_k not in (1, 2):
        print(f"error: k must be 1 or 2 (got {top_k})", file=sys.stderr)
        return 2
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else WORK / f"submission_task1_v3_k{top_k}.jsonl"

    if not META_FILE.exists():
        print(f"error: {META_FILE} not found - run the CV-median extraction first", file=sys.stderr)
        return 1

    meta = json.loads(META_FILE.read_text())
    labels = meta["labels"]
    per_class = meta["thresholds_per_class"]
    thresholds = np.array([per_class[l] for l in labels], dtype=np.float32)
    print(f"model: {MODEL_DIR.name}  thresholds source: {meta.get('thresholds_source', 'unknown')}")
    print(f"top_k: {top_k}  templates: templates_v3.py")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    test_rows = load_jsonl(DEFAULT_TEST)
    print(f"test cases: {len(test_rows)}  device: {device}  pool: {POOL}")

    facts = [r["fact"] for r in test_rows]
    print("chunked case-level inference...")
    probs, owners = predict_probs_chunked(model, tok, facts, device, pool=POOL)
    print(f"  {len(owners)} chunks / {len(facts)} cases")

    print("per-sentence scoring...")
    sent_scores = score_all_sentences_by_id(model, tok, test_rows, device)

    sub_rows = []
    section_counts = Counter()
    for r, pvec in zip(test_rows, probs):
        exp, reasoning = build_reasoning_and_explanation(
            pvec, labels, thresholds, r, sent_scores, top_k
        )
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
    print(f"\nexplanation entries per case: min={min(per_case_n)} "
          f"median={sorted(per_case_n)[len(per_case_n)//2]} "
          f"max={max(per_case_n)} mean={sum(per_case_n)/len(per_case_n):.2f}")
    per_case_secs = [sum(len(v) for v in s["explanation"].values()) for s in sub_rows]
    print(f"total section labels per case:  min={min(per_case_secs)} "
          f"median={sorted(per_case_secs)[len(per_case_secs)//2]} "
          f"max={max(per_case_secs)} mean={sum(per_case_secs)/len(per_case_secs):.2f}")
    reasoning_lens = [len(s["reasoning_traces"].split()) for s in sub_rows]
    print(f"reasoning_traces word len:      min={min(reasoning_lens)} "
          f"median={sorted(reasoning_lens)[len(reasoning_lens)//2]} "
          f"max={max(reasoning_lens)} mean={sum(reasoning_lens)/len(reasoning_lens):.1f}")

    if problems:
        print(f"\nWARNING: {len(problems)} problems")
        for p in problems[:15]:
            print(f"  {p}")
    else:
        print(f"\nOK: all {len(sub_rows)} rows validate")

    with out_path.open("w") as f:
        for r in sub_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}  ({out_path.stat().st_size:,} bytes)")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
