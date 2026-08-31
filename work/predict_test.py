"""Predict on the FIRE 2026 Task 01 test file and emit the submission JSONL.

Test schema (input):
    {"id": "ST-PRED-XXXX", "fact": "...", "reasoning_traces": null,
     "explanation": {"": []}}

Submission schema (output — same shape, null fields filled):
    {"id": "...", "fact": "...",
     "reasoning_traces": "step-by-step string",
     "explanation": {"verbatim sentence from fact": ["IPC XXX", ...], ...}}

Reuses predict_classifier.py building blocks: chunked mean-pool inference,
per-class thresholds, hybrid classifier/BM25 sentence selector. Adapts
label format (internal "Section 302 IPC" -> submission "IPC 302") and
output shape (explanation dict instead of statute list).

Verbatim recovery: sentence_split.py joins tokens with single spaces, so
outputs may not be substrings of the raw fact (newlines, double spaces,
tabs). We walk the fact character-by-character and locate the original
span by whitespace-stripped match, then emit the original substring.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_classifier import (
    predict_probs_chunked, score_all_sentences,
    bm25_select_sentence, section_num,
    SELECTOR_BY_SECTION, SECTION_HINTS, USE_CLASSIFIER_SELECT, POOL,
    tokenize as bm25_tokenize,
)
from templates import build_trace
from sentence_split import sentences as smart_sentences

import math
from collections import Counter


def bm25_best_with_score(fact_text: str, query: str,
                         k1: float = 1.5, b: float = 0.75):
    """Same math as predict_classifier.bm25_select_sentence but returns the
    best score too. Score == 0 means no keyword overlap: caller should fall
    back to the classifier rather than blindly grabbing sentence 0."""
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

    def score(i: int) -> float:
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

WORK = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = WORK / "model_bert"
DEFAULT_TEST = Path("/Users/nomitachetia/Desktop/rp/Task 1/task_1_statute_prediction_test.jsonl")
DEFAULT_OUT = WORK / "submission_task1.jsonl"


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open() if l.strip()]


def to_submission_label(internal: str) -> str:
    """Map 'Section 302 IPC' -> 'IPC 302' (per email spec)."""
    n = section_num(internal)
    return f"IPC {n}" if n else internal


def find_verbatim(sent: str, fact: str) -> str:
    """Return the original substring of `fact` that corresponds to `sent`.

    Sentences from sentence_split.py are token-rejoined with single spaces
    and may differ from the raw fact by whitespace only. This walks the
    fact keeping (char, original_index) for every non-whitespace char, then
    locates the sentence via a whitespace-stripped match.

    If nothing matches, returns `sent` unchanged. Callers should assert
    the result is a substring; verify_output() below does.
    """
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


def build_reasoning(section_sentence_pairs: list) -> str:
    """Concatenate one template output per predicted section, blank-line separated."""
    paras = []
    for num, sent in section_sentence_pairs:
        paras.append(build_trace(num, sent))
    return "\n\n".join(paras)


def predict_one_case(pvec: np.ndarray, labels: list, thresholds: np.ndarray,
                     row: dict, sent_scores: dict) -> tuple:
    """Return (explanation_dict, reasoning_traces_str) for a single test case."""
    order = list(np.argsort(-pvec))
    sel = [i for i in order if pvec[i] >= thresholds[i]]
    if not sel:
        sel = [int(order[0])]  # always emit at least one

    fact = row["fact"]
    # gather (section_num, sentence) — order preserved by predicted prob
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
            # else: fall through to classifier — better than first-sentence fallback
        if picked is None and sent_scores is not None:
            sents, sprobs = sent_scores[row["id"]]
            best_idx = int(np.argmax(sprobs[:, i]))
            picked = sents[best_idx]
        if picked is None:  # last-resort (no classifier sent scores)
            picked = smart_sentences(fact)[0] if smart_sentences(fact) else fact
        exact = find_verbatim(picked, fact)
        ordered.append((num, exact))

    # dedupe: if two sections landed on the same sentence, merge into list
    exp = {}
    for num, sent in ordered:
        submission = f"IPC {num}"
        if sent in exp:
            if submission not in exp[sent]:
                exp[sent].append(submission)
        else:
            exp[sent] = [submission]

    reasoning = build_reasoning(ordered)
    return exp, reasoning


def score_all_sentences_by_id(model, tok, test_rows, device):
    """Same as predict_classifier.score_all_sentences but keyed by 'id' not 'doc_id'."""
    from predict_classifier import SENT_BATCH, MIN_SENT_LEN, predict_probs
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


def verify_output(rows: list, sub_rows: list) -> tuple:
    """Return (n_ok, list_of_problems). Every explanation key must be a verbatim
    substring of its fact, every value must match 'IPC XXX' format, and every
    id must be present."""
    id2fact = {r["id"]: r["fact"] for r in rows}
    problems = []
    ipc_re = re.compile(r"^IPC \S+$")
    seen_ids = set()
    for s in sub_rows:
        seen_ids.add(s["id"])
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
            if not isinstance(sections, list) or not sections:
                problems.append(f"{s['id']}: value not a non-empty list")
                continue
            for lbl in sections:
                if not ipc_re.match(lbl):
                    problems.append(f"{s['id']}: bad label format {lbl!r}")
    for r in rows:
        if r["id"] not in seen_ids:
            problems.append(f"{r['id']}: missing from submission")
    return len(sub_rows) - len({p.split(':')[0] for p in problems}), problems


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    test_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TEST
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT

    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        return 1
    if not test_path.exists():
        print(f"ERROR: test file not found: {test_path}", file=sys.stderr)
        return 1

    meta = json.loads((model_dir / "meta.json").read_text())
    labels = meta["labels"]
    per_class = meta.get("thresholds_per_class")
    if per_class:
        thresholds = np.array([per_class[l] for l in labels], dtype=np.float32)
        print(f"model: {model_dir.name}  per-class thresholds present  "
              f"saved dev_F1={meta.get('tuned_dev_macro_f1', meta['best_dev_macro_f1']):.4f}")
    else:
        g = float(meta["threshold"])
        thresholds = np.full(len(labels), g, dtype=np.float32)
        print(f"model: {model_dir.name}  global threshold={g:.2f}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    test_rows = load_jsonl(test_path)
    print(f"test cases: {len(test_rows)}   device: {device}   pool: {POOL}")

    facts = [r["fact"] for r in test_rows]
    print("chunked case-level inference...")
    probs, owners = predict_probs_chunked(model, tok, facts, device, pool=POOL)
    n_chunks = len(owners)
    per_case_counts = [owners.count(i) for i in range(len(facts))]
    print(f"  {n_chunks} chunks / {len(facts)} cases "
          f"(min={min(per_case_counts)} median={sorted(per_case_counts)[len(per_case_counts)//2]} "
          f"max={max(per_case_counts)} chunks/case)")

    sent_scores = None
    if USE_CLASSIFIER_SELECT:
        print("per-sentence scoring for classifier-based selection...")
        sent_scores = score_all_sentences_by_id(model, tok, test_rows, device)

    sub_rows = []
    section_counts = {}
    for r, pvec in zip(test_rows, probs):
        exp, reasoning = predict_one_case(pvec, labels, thresholds, r, sent_scores)
        sub_rows.append({
            "id": r["id"],
            "fact": r["fact"],
            "reasoning_traces": reasoning,
            "explanation": exp,
        })
        for lbls in exp.values():
            for l in lbls:
                section_counts[l] = section_counts.get(l, 0) + 1

    n_ok, problems = verify_output(test_rows, sub_rows)
    print(f"\nsections predicted (across {len(sub_rows)} cases):")
    for l in sorted(section_counts, key=lambda k: -section_counts[k]):
        print(f"  {l}: {section_counts[l]}")

    per_case_n = [len(s["explanation"]) for s in sub_rows]
    print(f"\nexplanation entries per case: min={min(per_case_n)} "
          f"median={sorted(per_case_n)[len(per_case_n)//2]} "
          f"max={max(per_case_n)} mean={sum(per_case_n)/len(per_case_n):.2f}")
    per_case_secs = [sum(len(v) for v in s["explanation"].values()) for s in sub_rows]
    print(f"total section labels per case:  min={min(per_case_secs)} "
          f"median={sorted(per_case_secs)[len(per_case_secs)//2]} "
          f"max={max(per_case_secs)} mean={sum(per_case_secs)/len(per_case_secs):.2f}")

    if problems:
        print(f"\nWARNING: {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    else:
        print(f"\nOK: all {len(sub_rows)} rows validate against submission schema")

    with out_path.open("w") as f:
        for r in sub_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}  ({out_path.stat().st_size:,} bytes)")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
