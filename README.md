# FIRE 2026 Task 1 — Explainable Statute Prediction

Pipeline for the FIRE 2026 shared task **"LLM as a Judge? — From Statute Prediction to Sycophancy Detection in Law"**, Task 1 (Explainable Statute Prediction, ESP).

Given the factual narrative of an Indian Supreme Court case, the system predicts (a) the applicable IPC sections from the seven-class label set `{147, 201, 302, 376, 420, 498A, 506}`, (b) the verbatim sentence in the fact that triggers each predicted section, and (c) a free-text legal reasoning trace per case.

## System overview

```
raw fact text
     │
     ▼
[legal-aware sentence splitter]  ─► candidate sentences
     │
     ▼
[chunked InLegalBERT classifier] ─► per-section probabilities (mean-pool)
     │
     ▼
[per-class thresholds]           ─► section set
     │
     ▼
for each predicted section:
   ├─ [hybrid sentence selector] ─► triggering sentence (classifier or BM25)
   └─ [reasoning template]       ─► legal argument text
     │
     ▼
{id, fact, reasoning_traces, explanation}
```

Base encoder: [`law-ai/InLegalBERT`](https://huggingface.co/law-ai/InLegalBERT). Long-fact handling via sliding-window chunking (stride 384) with mean-pool over per-chunk probabilities. Per-class decision thresholds are tuned on a dev split. Sentence selection uses the same InLegalBERT classifier at the sentence level for §§147/201/302/420/498A and BM25 keyword search for §§376/506, with a classifier fallback when BM25 returns zero score.

## Results (5-fold CV, seed=42)

Trained on 4/5 folds (~420 cases), evaluated on the held-out fold (~105 cases), rotated:

| Metric | Mean ± Std | Min | Max |
|---|---|---|---|
| Macro F1 @ per-fold tuned thresholds | **0.781 ± 0.021** | 0.757 | 0.813 |
| Macro F1 @ frozen thresholds (v1 dev-tuned) | 0.724 ± 0.029 | 0.693 | 0.767 |
| Macro F1 @ global 0.5 threshold | 0.636 ± 0.046 | 0.578 | 0.690 |
| Recall @ 3 | **0.970 ± 0.008** | 0.962 | 0.981 |

Per-class F1 (tuned, 5-fold mean):

| Section | Mean | Std | Notes |
|---|---|---|---|
| §376 (rape) | 0.955 | 0.019 | Rock solid |
| §420 (cheating) | 0.955 | 0.030 | Rock solid |
| §498A (cruelty) | 0.911 | 0.036 | Strong |
| §302 (murder) | 0.786 | 0.043 | Strong |
| §506 (intimidation) | 0.643 | **0.161** | Unstable — see paper |
| §147 (rioting) | 0.618 | 0.094 | Weak |
| §201 (evidence-destruction) | 0.597 | 0.059 | Weak but stable |

Full per-fold breakdown in [`work/cv_5fold_results.json`](work/cv_5fold_results.json).

## Repository layout

```
work/
  ├── sentence_split.py           legal-aware sentence splitter (60+ abbrevs)
  ├── prepare.py                  data preparation + 80/20 split
  ├── baselines.py                majority + BM25-NN baselines
  ├── train_classifier.py         InLegalBERT trainer (420-case train)
  ├── train_classifier_chunked.py chunked-training variant (rejected)
  ├── train_full.py               retrain on all 525 cases
  ├── tune_thresholds.py          per-class threshold sweep
  ├── predict_classifier.py       core prediction + inference utilities
  ├── predict_test.py             adapter for the FIRE test schema
  ├── predict_test_v3.py          v3 predictor (CV-median thresholds + polished templates + top-K)
  ├── predict_ensemble.py         v1+v2 sigmoid ensembling
  ├── templates.py                original 7 per-section reasoning templates
  ├── templates_v3.py             polished templates (v3)
  ├── eval.py                     Macro F1 + Recall@3 harness
  ├── eval_exact_fact.py          sentence-match scoring
  ├── cv_5fold.py                 5-fold cross-validation runner
  ├── cv_5fold_results.json       per-fold metrics + summary
  └── compare_submissions.py      v1 vs v2 vs v3 diff tool
```

The trained model weights (`work/model_bert/` and `work/model_bert_full/`, each ~420 MB) are **not** in this repository — retrain via the scripts below.

## Reproducing

### Data (not included)

The training corpus is the **PROSLEX** dataset (525 Indian SC cases annotated with IPC section applicability). It is not included here — please obtain it from the authors as described in the paper below. Place `task1.jsonl` at the repository root.

> Adhikary, S., Bhattacharya, U., Singh, V. K., Sharma, A., Nigam, S. K., Das, S., Guha, S. K., Rudra, K., & Ghosh, K. (2026). *PROSLEX: A Novel Dataset for Expert-Annotated Legal Statute Prediction for Indian Judiciary.* [arXiv:2608.08830](https://arxiv.org/abs/2608.08830).

The FIRE test set (`task_1_statute_prediction_test.jsonl`, 57 cases) is distributed privately by the FIRE organizers to registered participants — not included.

### Setup

```bash
cd work
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers numpy
```

Tested with Python 3.13, torch 2.12.1, transformers 5.12.1, on Apple Silicon (MPS backend). Should work on CUDA with minor adjustments.

### Pipeline

```bash
# 1. Prepare data (produces train.jsonl, dev.jsonl at work/)
python prepare.py

# 2. Train the 420-case model with dev-driven threshold tuning
python train_classifier.py
python tune_thresholds.py

# 3. Retrain on all 525 cases (drops dev holdout)
python train_full.py

# 4. Predict on the test set (using the retrained 525-case model)
python predict_test.py model_bert_full path/to/test.jsonl submission.jsonl

# 5. (optional) 5-fold CV for a defensible mean-±-std F1 for the paper
python cv_5fold.py
```

### v3 variants

```bash
# Predict with CV-median thresholds + polished templates
python predict_test_v3.py 1     # top-1 sentence per section
python predict_test_v3.py 2     # top-2 sentences per section

# v1 + v2 sigmoid ensemble
python predict_ensemble.py
```

## Design decisions & negative results

- **Head-truncated training beat chunked training** (F1 0.804 → 0.783 on the 105-case dev split; chunked-labelled sub-sequences introduced label noise). See `train_classifier_chunked.py` — kept for the paper's ablation section.
- **Mean-pool beat max-pool** over chunk-level probabilities (F1 0.804 vs 0.764). Max-pool lets the noisiest chunk dominate.
- **Per-class thresholds beat a global 0.5** (F1 0.759 → 0.804 on the dev split; 5-fold CV shows the general lift is +0.057).
- **BM25 fallback to classifier** eliminated 10 first-sentence artefacts for §376/§506 when the BM25 hint returned zero score.

## Known limitations

- **Closed 7-section vocabulary** — the test set contains IPC 34/149/120B/304/304B/307/313/366 in ~24 cases. This pipeline cannot predict them.
- **§506 (intimidation) F1 is unstable across CV folds** (std 0.161). Threat language in Indian judgments is inconsistent; the BM25 hint helps but does not fully compensate.
- **§201 (evidence-destruction) is an auxiliary charge** and lacks distinctive lexical signal — it rides along with the principal offence.
- **Legal Semantic Score encoder is undisclosed** by the FIRE organisers, so it cannot be optimised against directly.

## Citation

If you use this code, please cite the PROSLEX dataset paper:

```bibtex
@misc{adhikary2026proslexnoveldatasetexpertannotated,
  title  = {PROSLEX: A Novel Dataset for Expert-Annotated Legal Statute Prediction for Indian Judiciary},
  author = {Subinay Adhikary and Upal Bhattacharya and Vivek Kumar Singh and
            Anurag Sharma and Shubham Kumar Nigam and Suvasis Das and
            Shouvik Kumar Guha and Koustav Rudra and Kripabandhu Ghosh},
  year   = {2026},
  eprint = {2608.08830},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url    = {https://arxiv.org/abs/2608.08830}
}
```

## License

MIT — see [LICENSE](LICENSE).
