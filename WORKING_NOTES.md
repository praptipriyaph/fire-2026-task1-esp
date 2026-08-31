# Explainable IPC Section Prediction via Fine-tuned InLegalBERT, Long-Fact Chunking, and Templated Legal Reasoning

**FIRE 2026 — Task 01: Explainable Statute Prediction**

**Author:** *[your name]*
**Affiliation:** *[your affiliation]*
**Date:** *[draft — 2026-06-22; final due 2026-08-20]*

---

## Abstract

We describe a three-stage pipeline for predicting applicable Indian Penal Code (IPC) sections from Supreme Court case facts, locating the supporting sentence in the facts, and generating a legal reasoning trace. The system fine-tunes InLegalBERT for multi-label classification across the seven IPC sections present in the training corpus (147, 201, 302, 376, 420, 498A, 506); applies sliding-window chunked inference with mean-pooling to recover signal from the ~47 % of case facts that exceed the model's 512-token window; tunes per-class decision thresholds on a held-out dev split; selects the exact supporting sentence per predicted section via a hybrid classifier/BM25 strategy operating over a legal-abbreviation-aware sentence splitter; and emits a reasoning trace from per-section legal templates that cite the substantive offence-defining section. On a held-out 105-case dev split (20% of the 525-case training release), the final pipeline achieves **Macro F1 = 0.793** and **Recall@3 = 0.957** on section labels, with **51.8 % of selected supporting sentences exactly matching the gold annotation** for correctly predicted sections. We also report a negative result: re-training the classifier on chunked inputs (one training example per chunk, inheriting the case's labels) consistently underperforms the baseline of *truncated training + chunked inference*, because chunk-level label inheritance is noisy when individual chunks contain no evidence for some of the case's labels.

---

## 1. Introduction

Task 01 of FIRE 2026 ("Explainable Statute Prediction", ESP) requires three structured outputs per case: (i) the set of applicable IPC sections, (ii) the verbatim sentence(s) from the case facts that trigger each predicted section, and (iii) a legal reasoning trace connecting the facts to the statute. The evaluation is composite — Macro F1 on section labels (35 %), ROUGE-L (25 %) and BLEU (20 %) on reasoning traces, Recall@3 on section labels (10 %), and a held-out Legal Semantic Score (10 %) — placing the majority of the score on reasoning quality.

Our system separates the three outputs into independently optimisable stages, with the classifier supplying signal to both the section-label output and the sentence-selection output. Reasoning generation is treated as a constrained template-filling problem, given the absence of any gold reasoning supervision in the training release.

## 2. Data

The released training set (`task1.jsonl`) contains 525 Indian Supreme Court cases. Each record provides:

- `doc_id`: case identifier (e.g. `2013.INSC.228.txt`)
- `fact`: full factual description (~700 words on average, up to ~5 000 tokens)
- `statute`: list of applicable IPC sections as strings (e.g. `"IPC 147"`)
- `explanation`: a dictionary mapping each supporting sentence (verbatim from `fact`) to its corresponding IPC section

Notably, **no `reasoning_trace` field is provided** in the released training data. The submission schema (per §06 of the task page) expects each statute entry to take the form `{section, exact_fact, reasoning_trace}`, which we construct from the `explanation` dictionary and per-section templates.

### 2.1 Label distribution

The training set covers exactly seven IPC sections with no long tail and no singletons. The distribution is roughly balanced relative to typical legal-NLP datasets:

| Section | Offence | Count |
|---|---|---:|
| 302 | Murder | 181 |
| 498A | Cruelty by husband/relative | 84 |
| 376 | Rape | 83 |
| 420 | Cheating | 80 |
| 147 | Rioting | 77 |
| 506 | Criminal intimidation | 65 |
| 201 | Causing disappearance of evidence | 51 |

Mean sections per case is 1.18 (range 1–3); mean number of supporting sentences per case in `explanation` is 4.13. Approximately 88 % of `explanation` keys are exact substrings of the corresponding `fact`.

### 2.2 Fact length

Tokenised with the InLegalBERT tokenizer, the mean fact length is 629 tokens (median 466, max 5 060). **46.7 % of training facts and 46.7 % of dev facts exceed the model's 512-token maximum input length**, motivating the chunked-inference strategy described in §3.1.2. About 14 % of facts exceed 1 024 tokens and 3 % exceed 2 048.

### 2.3 Schema normalisation

The local schema uses `"IPC X"` while the submission spec uses `"Section X IPC"`. We normalise via a regex (`IPC\s+(\S+)` → `Section \1 IPC`) before all downstream stages. Statute entries are produced one per `(section, supporting-sentence)` pair drawn from `explanation`.

### 2.4 Train/dev split

We hold out 20 % of cases (n = 105) as a dev split for threshold tuning and selector evaluation, using `random.seed(42)`. The remaining 420 cases form the training set. Every section retains at least 8 dev cases; the split is unstratified due to its small size.

## 3. System

### 3.1 Output 1 — Section classifier

#### 3.1.1 Model and training

We fine-tune `law-ai/InLegalBERT` (BERT-base, 110 M parameters, pre-trained on Indian legal text) as a multi-label classifier with a 7-way sigmoid head and BCEWithLogitsLoss (set via `problem_type="multi_label_classification"`). Optimisation uses AdamW (lr = 2 × 10⁻⁵), batch size 8, max sequence length 512, 10 epochs, no warmup, no weight decay, seed = 42. Each training case is fed to the model **head-truncated to 512 tokens**. After each epoch we evaluate dev Macro F1 sweeping a global threshold in [0.20, 0.70] (step 0.05) and persist the best checkpoint. Training is performed on a single Apple M-series GPU via the Metal Performance Shaders (MPS) backend; one epoch over 420 examples takes ~75 seconds.

#### 3.1.2 Long-fact handling via chunked inference

Because 47 % of facts exceed 512 tokens, head-truncation discards meaningful evidence — particularly for offences (498A cruelty, 147 unlawful-assembly narratives) whose supporting events often appear after extensive procedural preamble. At inference time we therefore split every fact into overlapping windows of 510 tokens with stride 384 (128-token overlap), prepend `[CLS]` and append `[SEP]` to each window, and run the model on the resulting batch. Per-class probabilities are **mean-pooled across the chunks of each case** before applying decision thresholds. On dev, this raises Macro F1 from 0.781 (truncated single-pass) to 0.793 and Recall@3 from 0.943 to 0.957. We explicitly compared mean-pool to max-pool: max-pool *hurts* F1 (0.764 after retuning) because it lets the most aggressive chunk dominate per class, surfacing spurious predictions from chunks that contain no evidence for the predicted section. Mean-pool dilutes such noise across all chunks while still admitting tail-only evidence into the final probability.

#### 3.1.3 Per-class threshold tuning

With the best checkpoint loaded and chunked + mean-pool inference active, we sweep a per-section threshold in [0.10, 0.70] (step 0.05) independently for each class, optimising per-class F1 on the dev split. The tuned thresholds (0.10 – 0.50 across the 7 classes) replace the single global threshold at deployment time. To prevent empty predictions, we always emit at least one section per case (the argmax over predicted probabilities, regardless of threshold).

### 3.2 Output 2 — Supporting sentence selection

For each predicted section we emit a verbatim sentence from `fact` that triggers that section.

#### 3.2.1 Legal-aware sentence splitter

A punctuation-based regex (`(?<=[.!?])\s+`) misclassifies common legal-writing patterns — abbreviated titles (*"Mr. Singh"*, *"Hon'ble J. Krishna"*), numerical-suffix references (*"accused no. 2"*, *"PW-1"*), code references (*"Sec. 302 read with Sec. 34"*), reporter citations (*"A.I.R. 1957 S.C. 432"*) — as sentence boundaries, fragmenting the candidate-sentence pool and degrading downstream selection. We therefore implement a custom splitter that walks token by token: a token ending in `.!?` is suppressed as a sentence boundary when (i) it equals one of a curated list of 60+ legal abbreviations (titles, code parts, reporter abbreviations, Latin shorthand), (ii) it contains internal dots indicating an inline acronym (e.g. *"A.I.R."*), (iii) it is a single uppercase letter (initial), (iv) the next token starts with a digit or a lowercase letter, or (v) it is a witness identifier matching `(P|D|C)W-?\d+`. The splitter is implemented in `sentence_split.py` and improves the dev strict `exact_fact` match rate from 0.468 to 0.505 in isolation.

#### 3.2.2 Classifier-based and BM25-hint selectors

**Classifier-based.** Every candidate sentence from the splitter is passed through the same fine-tuned classifier (batch size 32, max length 128). For each predicted section, the sentence with the highest classifier probability for that section is selected.

**BM25-hint.** For each IPC section we author a short keyword query capturing the offence vocabulary (e.g. for §302: *"murder death killed shot stabbed weapon"*). Okapi BM25 scores each candidate sentence against the query; the top-scoring sentence is selected.

#### 3.2.3 Hybrid per-section selection

On dev the two strategies are complementary: the classifier-based selector dominates for §§ 147 / 201 / 302 / 420 / 498A while BM25 dominates for §§ 376 / 506. The latter pair are offences whose evidentiary sentence typically contains the offence keyword directly (e.g. *"threatened to kill"* for §506, *"raped"* for §376), which BM25 picks up reliably, while the classifier is sometimes distracted by adjacent narrative sentences. The hybrid selector applies whichever method wins per section.

### 3.3 Output 3 — Templated reasoning

#### 3.3.1 Constraint

The released training data contains no `reasoning_trace` labels. We cannot fine-tune or even evaluate ROUGE-L / BLEU locally. The single published example in the task description (§04) is the only ground truth visible at design time.

#### 3.3.2 Template structure

We hand-craft one template per IPC section. Each template follows a three-sentence structure derived from the published example:

> *"Section X IPC applies because [statement of legal ingredients of the offence]. {exact_fact}. The presence of these facts satisfies the elements of [offence name] under Section Y IPC, making Section X the appropriate penal provision."*

where Section Y is the substantive offence-defining section (e.g. §300 for §302, §375 for §376, §141 / 146 for §147, §503 for §506). The `{exact_fact}` slot is filled with the verbatim sentence selected at stage 3.2, so its surface tokens contribute directly to ROUGE/BLEU overlap with whatever gold reasoning references the same sentence.

This design optimises for surface-form overlap (ROUGE-L, BLEU) and legal-vocabulary similarity (LSS) without requiring training labels. It is interpretable, deterministic, and easy to revise per-section once any gold reasoning becomes visible.

## 4. Results on dev (n = 105)

### 4.1 Cumulative pipeline progression

| Configuration | Macro F1 | Recall@3 | `exact_fact` strict match |
|---|---:|---:|---:|
| Majority class (always §302) | 0.079 | 0.302 | — |
| BM25 nearest-neighbour (train fact retrieval) | 0.491 | 0.546 | — |
| InLegalBERT, 5 epochs, head-truncated | 0.697 | 0.952 | — |
| InLegalBERT, 10 epochs, head-truncated | 0.753 | 0.943 | — |
| + per-class thresholds | 0.781 | 0.943 | 0.349 |
| + classifier-based sentence selector | 0.781 | 0.943 | 0.450 |
| + hybrid (per-section) selector | 0.781 | 0.943 | 0.468 |
| + legal-aware sentence splitter | 0.781 | 0.943 | 0.505 |
| **+ chunked inference (mean-pool, retuned)** | **0.793** | **0.957** | **0.518** |

### 4.2 Per-class F1 (final configuration)

| Section | P | R | F1 | Tuned threshold |
|---|---:|---:|---:|---:|
| 420 (cheating) | 1.00 | 1.00 | **1.000** | 0.10 |
| 376 (rape) | 0.85 | 0.92 | **0.880** | 0.50 |
| 498A (cruelty) | 0.79 | 1.00 | **0.884** | 0.15 |
| 302 (murder) | 0.76 | 0.95 | 0.844 | 0.50 |
| 147 (rioting) | 0.65 | 0.77 | 0.703 | 0.35 |
| 506 (intimidation) | 0.62 | 0.73 | 0.667 | 0.20 |
| 201 (evidence) | 0.46 | 0.75 | 0.571 | 0.20 |

§420 reaches a perfect F1 of 1.000 on dev under this configuration. The remaining headroom lies in §§ 201 / 506, which are typically charged in addition to a principal offence and whose evidentiary sentences are dispersed across the fact narrative.

### 4.3 Sentence-selection quality

Strict `exact_fact` match rate, scored on the subset of predictions where the section is correct (n = 112 of 150 final-pipeline predicted entries):

| Selector | Strict match rate |
|---|---:|
| BM25-hint only | 0.349 |
| Classifier-based only | 0.450 |
| Hybrid (per-section) | 0.468 |
| Hybrid + legal-aware splitter | 0.505 |
| **Final pipeline (hybrid + splitter + chunked classifier)** | **0.518** |

Per-section strict match rates under the final pipeline span 0.12 – 0.82, with §498A highest (0.82) and §420 lowest (0.12). The §420 anomaly persists even after sentence-splitter improvements: cheating-case evidentiary sentences are typically *specific* transactional narratives (account numbers, dates, amounts) that share little surface form with the offence keyword "cheating", confusing both the classifier-based selector (which uses the same model that classifies on holistic facts, not transactional detail) and the BM25-hint selector (whose hint vocabulary is offence-generic).

### 4.4 Reasoning quality

We are unable to score ROUGE-L, BLEU, or LSS locally — these require gold reasoning traces that are not present in the released training data. Reasoning quality is therefore optimised structurally, by:

- mirroring the three-sentence form of the published example;
- terminating each trace with the published example's closing phrase ("making Section X the appropriate penal provision") for predictable n-gram overlap;
- citing the substantive section (§ 300, § 375, § 141 / 146, § 503) that legal prose conventionally references when justifying the corresponding penal section;
- injecting the verbatim `exact_fact` as a standalone sentence so its tokens count toward overlap.

### 4.5 Negative result: chunked training

A natural extension of §3.1.2 is to *train* the classifier on chunked inputs as well: for each training case, generate the same sliding-window chunks and feed each as an independent training example, inheriting the case's multi-hot label set. This expands the training pool from 420 cases to 792 chunks (avg 1.89 chunks per case).

We trained for 10 epochs with otherwise identical hyperparameters; best dev tuned Macro F1 was reached at epoch 5, dev metrics fluctuated thereafter, and the train loss continued to fall to 0.030 (vs 0.08 for the head-truncated trainer over 10 epochs) — consistent with the chunked classifier overfitting faster on the chunk-level noisy supervision.

| Configuration | Macro F1 | Recall@3 | `exact_fact` strict match |
|---|---:|---:|---:|
| Truncated training + chunked inference | **0.793** | 0.957 | **0.518** |
| Chunked training + chunked inference | 0.783 | **0.983** | 0.464 |

Chunked training improves Recall@3 (gains on §§ 147 / 376 / 506) but *degrades* both Macro F1 and `exact_fact` match. The degradation is intuitive: when a chunk contains evidence for only a subset of the case's labels, it is still trained against the full label set, teaching the model to fire those labels even on irrelevant chunks. Two downstream consequences follow: (i) per-chunk probabilities are pushed up across all classes at inference, hurting precision after mean-pooling; and (ii) the same classifier is used to score sentences for `exact_fact` selection — a classifier that is *less* discriminative about which chunk contains which evidence is also a worse sentence selector. We therefore retain truncated training as our deployed configuration and report this as a negative result.

## 5. Lower-bound composite estimate

Combining the dev numbers we *can* measure:

| Component | Weight | Dev value | Contribution |
|---|---:|---:|---:|
| Macro F1 | 0.35 | 0.793 | 0.278 |
| Recall@3 | 0.10 | 0.957 | 0.096 |
| ROUGE-L | 0.25 | — | — |
| BLEU | 0.20 | — | — |
| LSS | 0.10 | — | — |
| **Measurable lower bound** | **0.45** | | **0.374** |

i.e. on label-driven metrics alone the system achieves 37.4 of a possible 45 composite points on dev (~83 % of the label-driven maximum).

## 6. Limitations

- **No gold reasoning supervision** in the released training set. All reasoning-quality optimisation is a one-shot exercise against the single published example; we cannot iterate on templates against ROUGE/BLEU because we cannot compute either locally.
- **Single random dev split** with no cross-validation. With only 8 – 40 cases per class in dev, per-class threshold estimates carry non-trivial variance.
- **Dev-tuned hyperparameters** — per-class thresholds and the hybrid selector map — fit dev and may not generalise to the test set's section distribution if it differs materially.
- **Vocabulary mismatch risk for novel sections.** The seven training sections are all from the Indian Penal Code criminal-offences chapter. If the test set introduces unseen sections, the classifier cannot predict them and templates do not exist.
- **`exact_fact` selection for §420** (cheating) remains poor (0.12 strict match) regardless of selector strategy, attributable to vocabulary mismatch between the offence keyword and the actual transactional evidentiary sentences. No clean fix is implemented.

## 7. Future work

Listed in roughly decreasing expected payoff per unit of effort.

1. **LLM-generated reasoning conditioned on `(fact, section, exact_fact)`.** Once the test set surfaces actual gold reasoning patterns (or FIRE releases additional examples), fine-tuning or few-shot prompting an open-weight LLM (e.g. Llama 3 / 8 B with QLoRA) on `(case, statute, generated-trace)` triples may close the structural-template gap on ROUGE-L and BLEU. This addresses the 55 % of the composite score we currently optimise blindly.
2. **5-fold cross-validation** for threshold and selector selection, reducing the variance of dev-tuned hyperparameters and producing a more honest estimate of test performance.
3. **Train on the full 525 cases for the final submission.** Once the configuration is locked, the held-out dev split is no longer needed; retraining on all available data should add roughly 0.01 – 0.02 to Macro F1 for free.
4. **Auxiliary-offence-aware training.** Oversample cases containing §§ 201 / 506 or condition the classifier on the principal offence to lift the two still-weak classes.
5. **Per-section pooling strategy.** Our chunked inference uses a single pool function (mean) across all classes. Per-class movement under max-pool versus mean-pool suggests §§ 498A / 147 prefer evidence-pooling-favouring max while §§ 201 / 376 prefer mean. A per-class pool could squeeze additional F1, at the cost of further dev overfitting.
6. **§420 sentence selector.** Cheating cases need a selector tuned to transactional vocabulary (amounts, accounts, dates) rather than the offence keyword.

## 8. Conclusion

We presented a pragmatic, fully reproducible pipeline for the FIRE 2026 ESP task that combines fine-tuned InLegalBERT classification with chunked-inference long-fact handling, per-class threshold calibration, a legal-aware sentence splitter feeding a hybrid sentence selector, and template-based reasoning generation. Despite the absence of gold reasoning supervision and a small (525-case) training release, the system reaches **0.793 Macro F1 / 0.957 Recall@3 / 0.518 strict exact-sentence match** on a held-out dev split, contributing a measurable 0.374 to the composite score from label-driven metrics alone. We also report a negative result for the natural extension of chunked training: chunk-level label inheritance is noisy and degrades both classification precision and downstream sentence selection. The main open question is how the templated reasoning traces score against the held-out gold rubric.

## References

- Paul, S., Mandal, A., Goyal, P., Ghosh, S. (2023). *Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law*. ICAIL. (InLegalBERT, `law-ai/InLegalBERT`.)
- Robertson, S. E., & Walker, S. (1994). *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval*. SIGIR. (BM25.)
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- Pappagari, R., Zelasko, P., Villalba, J., Carmiel, Y., Dehak, N. (2019). *Hierarchical transformers for long document classification*. ASRU. (Sliding-window + pooling baseline for long-document BERT inference.)
- FIRE 2026 Task 01 — *Explainable Statute Prediction.* Track page, June 2026.
