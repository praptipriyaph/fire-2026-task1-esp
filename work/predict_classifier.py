"""Predict with the trained InLegalBERT classifier and emit submission JSONL.

For each predicted section we still need:
  - exact_fact:      a verbatim sentence from the dev case's `fact`.
                     Strategy: run every candidate sentence through the same
                     classifier and pick the one with the highest predicted
                     probability for that section. This reuses what the model
                     already learned about section -> evidence and beats the
                     BM25-hint heuristic (which scored 35% match).
  - reasoning_trace: built from templates.build_trace(section_num, exact_fact).

A BM25-hint fallback is kept for cases where the classifier returns
near-uniform probabilities (no salient sentence). Set USE_CLASSIFIER_SELECT=False
to switch back to the old heuristic for comparison.

Also emits ranked top-3 sections per case (by predicted probability) so
Recall@3 is scored on the model's own ranking, not just the thresholded set.

Output: work/preds_bert.jsonl
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
from templates import build_trace
from sentence_split import sentences as smart_sentences

WORK = Path(__file__).resolve().parent
MODEL_DIR = WORK / "model_bert"
DEV = WORK / "dev.jsonl"
OUT = WORK / "preds_bert.jsonl"

MAX_LEN = 512
BATCH = 8
SENT_BATCH = 32                       # batch size for sentence-level scoring
USE_CLASSIFIER_SELECT = True          # False -> all BM25 hint
MIN_SENT_LEN = 5                      # skip very short candidates (e.g. "Held.")
CHUNK_FACTS = True                    # sliding window over long facts at inference
CHUNK_STRIDE = 384                    # 128-token overlap between chunks
POOL = "mean"                         # "max" or "mean" over chunk probs

# Per-section selector override, tuned on dev:
# classifier > BM25-hint for 147/201/302/420/498A; BM25-hint > classifier for 376/506.
SELECTOR_BY_SECTION = {
    "147":  "classifier",
    "201":  "classifier",
    "302":  "classifier",
    "376":  "bm25",
    "420":  "classifier",
    "498A": "classifier",
    "506":  "bm25",
}

SENT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[a-z]+")
SEC_NUM_RE = re.compile(r"Section\s+(\S+)\s+IPC")

# Short legal-context phrase per IPC section, used to bias sentence selection.
# These are NOT scored — only used internally as a query against case sentences.
SECTION_HINTS = {
    "147": "rioting unlawful assembly armed weapon force violence",
    "201": "evidence disappear destroy conceal screen offender",
    "302": "murder death killed shot stabbed weapon",
    "376": "rape sexual assault prosecutrix consent",
    "420": "cheating dishonestly deliver property fraud",
    "498A": "cruelty husband dowry harassment matrimonial",
    "506": "threat intimidation alarm harm",
}


def tokenize(s: str) -> list:
    return TOKEN_RE.findall(s.lower())


def sentences(text: str) -> list:
    return smart_sentences(text)


def section_num(label: str):
    m = SEC_NUM_RE.match(label)
    return m.group(1) if m else None


def bm25_select_sentence(fact_text: str, query: str,
                         k1: float = 1.5, b: float = 0.75) -> str:
    """Pick the sentence from `fact_text` with the highest BM25 score vs `query`."""
    sents = sentences(fact_text)
    if not sents:
        return fact_text.strip()
    docs = [tokenize(s) for s in sents]
    N = len(docs)
    dl = [len(d) for d in docs]
    avgdl = sum(dl) / N if N else 0.0
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    tf = [Counter(d) for d in docs]
    q_toks = tokenize(query)

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

    best_i = max(range(N), key=score)
    # if no token overlap, BM25 returns 0 for all - fall back to first sentence
    if score(best_i) == 0:
        return sents[0]
    return sents[best_i]


@torch.no_grad()
def predict_probs(model, tok, texts, device, batch_size=BATCH, max_len=MAX_LEN):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        all_probs.append(torch.sigmoid(out.logits).cpu().numpy())
    return np.concatenate(all_probs)


@torch.no_grad()
def predict_probs_chunked(model, tok, texts, device,
                          max_len=MAX_LEN, stride=CHUNK_STRIDE,
                          batch_size=BATCH, pool=POOL):
    """Long-fact inference via sliding-window chunks pooled per case.

    Texts that fit in `max_len` are predicted once (single chunk). Longer
    texts are split into overlapping chunks of (max_len - 2) tokens with
    `stride` step. Per-class probs are pooled across chunks (max or mean).

    Returns array shape [len(texts), n_labels].
    """
    chunk_size = max_len - 2  # room for [CLS] [SEP]
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    pad_id = tok.pad_token_id

    all_ids, all_attn, owners = [], [], []
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            ids = [tok.unk_token_id]
        if len(ids) <= chunk_size:
            starts = [0]
        else:
            starts = list(range(0, len(ids) - chunk_size + 1, stride))
            if starts[-1] + chunk_size < len(ids):
                starts.append(len(ids) - chunk_size)
        for s in starts:
            chunk = ids[s:s + chunk_size]
            seq = [cls_id] + chunk + [sep_id]
            attn = [1] * len(seq)
            pad_n = max_len - len(seq)
            if pad_n > 0:
                seq = seq + [pad_id] * pad_n
                attn = attn + [0] * pad_n
            all_ids.append(seq)
            all_attn.append(attn)
            owners.append(i)

    n_chunks = len(all_ids)
    n_labels = model.config.num_labels
    cp = np.zeros((n_chunks, n_labels), dtype=np.float32)
    for b in range(0, n_chunks, batch_size):
        in_ids = torch.tensor(all_ids[b:b + batch_size]).to(device)
        in_attn = torch.tensor(all_attn[b:b + batch_size]).to(device)
        logits = model(input_ids=in_ids, attention_mask=in_attn).logits
        cp[b:b + batch_size] = torch.sigmoid(logits).cpu().numpy()

    out = np.zeros((len(texts), n_labels), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.int32)
    if pool == "max":
        out -= 1.0  # so first chunk's value always wins for max
        for ci, owner in enumerate(owners):
            out[owner] = np.maximum(out[owner], cp[ci])
            counts[owner] += 1
        # any owner with 0 chunks would be -1; replace with zeros
        out[counts == 0] = 0.0
    elif pool == "mean":
        for ci, owner in enumerate(owners):
            out[owner] += cp[ci]
            counts[owner] += 1
        for o in range(len(texts)):
            if counts[o] > 0:
                out[o] /= counts[o]
    else:
        raise ValueError(f"unknown pool: {pool}")
    return out, owners


def score_all_sentences(model, tok, dev_rows, labels, device):
    """Return {doc_id: (sentences, probs[n_sents, n_labels])}.

    Sentences shorter than MIN_SENT_LEN tokens are filtered out (skip
    'Held.', etc.). If no sentence survives filtering, fall back to all.
    """
    # collect sentences per doc with offsets so we can re-split after batching
    per_doc_sents = {}
    flat_sents = []
    offsets = []  # (doc_id, start, end) into flat_sents
    for r in dev_rows:
        sents = sentences(r["fact"])
        kept = [s for s in sents if len(s.split()) >= MIN_SENT_LEN]
        if not kept:
            kept = sents or [r["fact"].strip()]
        start = len(flat_sents)
        flat_sents.extend(kept)
        end = len(flat_sents)
        per_doc_sents[r["doc_id"]] = kept
        offsets.append((r["doc_id"], start, end))

    probs = predict_probs(model, tok, flat_sents, device,
                          batch_size=SENT_BATCH, max_len=128)

    out = {}
    for did, s, e in offsets:
        out[did] = (per_doc_sents[did], probs[s:e])
    return out


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        print("Run train_classifier.py first.", file=sys.stderr)
        return 1

    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    labels = meta["labels"]
    global_thresh = float(meta["threshold"])
    per_class = meta.get("thresholds_per_class")  # {label: thresh} or None
    if per_class:
        thresholds = np.array([per_class[l] for l in labels], dtype=np.float32)
        print(f"loaded model: labels={len(labels)}  "
              f"per-class thresholds present  "
              f"saved dev_F1={meta.get('tuned_dev_macro_f1', meta['best_dev_macro_f1']):.4f}")
    else:
        thresholds = np.full(len(labels), global_thresh, dtype=np.float32)
        print(f"loaded model: labels={len(labels)}  threshold={global_thresh:.2f}  "
              f"saved dev_F1={meta['best_dev_macro_f1']:.4f}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    dev_rows = [json.loads(l) for l in DEV.open() if l.strip()]
    print(f"predicting on {len(dev_rows)} dev cases (device={device})  "
          f"selector={'classifier' if USE_CLASSIFIER_SELECT else 'bm25-hint'}  "
          f"chunked={CHUNK_FACTS}")

    facts = [r["fact"] for r in dev_rows]
    if CHUNK_FACTS:
        probs, owners = predict_probs_chunked(model, tok, facts, device, pool=POOL)
        n_chunks = len(owners)
        n_long = sum(1 for o in set(owners) if owners.count(o) > 1)
        print(f"  chunked inference: {n_chunks} chunks across "
              f"{len(facts)} cases ({n_long} required >1 chunk)")
    else:
        probs = predict_probs(model, tok, facts, device)

    sent_scores = None
    if USE_CLASSIFIER_SELECT:
        print("scoring per-sentence probabilities (one pass over all dev sentences)")
        sent_scores = score_all_sentences(model, tok, dev_rows, labels, device)

    preds = []
    for r, pvec in zip(dev_rows, probs):
        # ranked sections by prob (for top3_sections)
        order = list(np.argsort(-pvec))
        ranked = [labels[i] for i in order]

        # thresholded set (per-class), but always emit at least 1 (highest prob)
        sel = [i for i in order if pvec[i] >= thresholds[i]]
        if not sel:
            sel = [int(order[0])]

        statute = []
        for i in sel:
            sec = labels[i]
            num = section_num(sec)
            mode = SELECTOR_BY_SECTION.get(num, "classifier") \
                if USE_CLASSIFIER_SELECT else "bm25"
            if mode == "classifier" and sent_scores is not None:
                sents, sprobs = sent_scores[r["doc_id"]]
                best_idx = int(np.argmax(sprobs[:, i]))
                ef = sents[best_idx]
            else:
                hint = SECTION_HINTS.get(num, "")
                ef = bm25_select_sentence(r["fact"], hint)
            statute.append({
                "section": sec,
                "exact_fact": ef,
                "reasoning_trace": build_trace(num, ef),
            })
        preds.append({
            "doc_id": r["doc_id"],
            "statute": statute,
            "top3_sections": ranked[:3],
        })

    with OUT.open("w") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
