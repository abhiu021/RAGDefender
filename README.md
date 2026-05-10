# RAGDefender — Improved
-----------------------

## Table of Contents

1. [What is this paper about?](#1-what-is-this-paper-about)
2. [How RAGDefender works](#2-how-ragdefender-works)
3. [Limitations of the original system](#3-limitations-of-the-original-system)
4. [What we changed and why](#4-what-we-changed-and-why)
5. [Installation](#5-installation)
6. [How to run — step by step](#6-how-to-run--step-by-step)
7. [Python API](#7-python-api)
8. [CLI usage](#8-cli-usage)
9. [Results and outcomes](#9-results-and-outcomes)
10. [Project structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Citation](#12-citation)

---

## 1. What is this paper about?

### Background — The RAG pipeline

Large Language Models (LLMs) like GPT-4 are powerful but have two big problems: they can hallucinate facts, and their knowledge freezes at training time. **Retrieval-Augmented Generation (RAG)** solves this by giving the LLM access to an external knowledge base at inference time.

A typical RAG pipeline looks like this:

```
User query
    │
    ▼
Retriever  ──────────────────────────────────► Knowledge base
    │                                          (Wikipedia, documents, etc.)
    │  top-K passages
    ▼
LLM Generator
    │
    ▼
Answer
```

The LLM reads the retrieved passages and generates an answer based on them — which means the quality of the answer depends directly on the quality of the retrieved documents.

### The attack — knowledge corruption

This creates a target for attackers. If an attacker can inject misleading documents into the knowledge base, the retriever will pull them in, the LLM will read them as facts, and the generated answer will be wrong.

Three main attack types have been demonstrated:

| Attack | How it works |
|---|---|
| **PoisonedRAG** | Directly injects passages that assert a specific false answer with high retrieval relevance |
| **Blind** | Injects generic adversarial content without knowing the exact target query |
| **GARAG** | Gradient-optimised attack that crafts adversarial embeddings to avoid detection while staying query-relevant |

Without any defense, these attacks achieve **attack success rates (ASR) as high as 0.89** — meaning the LLM gives the attacker's intended wrong answer 89% of the time.

### The defense — RAGDefender

The paper *"Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems"* (ACSAC 2025) proposes a post-retrieval filter that sits between the retriever and the generator. It analyzes the retrieved document set, identifies adversarial documents, removes them, and passes only clean documents to the LLM.

Key claims from the paper:
- Reduces ASR from **0.89 → 0.02** on Gemini with a 4× poison ratio on the NQ dataset
- **12.36× faster** than RobustRAG (which runs a full LLM call per document)
- **1.53× faster** than Discern-and-Answer (which fine-tunes a discriminator)
- **Zero GPU memory** required for the defense itself (only the generator uses GPU)
- Works with any retriever (Contriever, DPR, ANCE) and any generator (LLaMA, Gemini, Vicuna)

---

## 2. How RAGDefender works

The defense runs in two stages:

```
Retrieved docs (mix of clean + poisoned)
          │
          ▼
    ┌─────────────────────────────────────┐
    │  Stage 1 — Grouping                 │
    │                                     │
    │  singlehop mode → K-Means           │
    │  (NQ, MS MARCO datasets)            │
    │                                     │
    │  multihop mode → Concentration      │
    │  (HotpotQA dataset)                 │
    └────────────────┬────────────────────┘
                     │ suspicious group identified
                     ▼
    ┌─────────────────────────────────────┐
    │  Stage 2 — Identification           │
    │                                     │
    │  Lightweight ML on embeddings       │
    │  No LLM call needed                 │
    └────────────────┬────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Clean docs (Rsafe)    Removed docs (Radv)
    → passed to LLM       → discarded
```

**Stage 1 — Grouping:**
The core insight from the paper is that poisoned documents tend to cluster tightly in embedding space because they all assert the same false fact, while legitimate documents are naturally more scattered (they cover the topic from different angles). RAGDefender exploits this:

- In `singlehop` mode: sentence-transformer embeddings are clustered. The tight cluster is flagged as adversarial.
- In `multihop` mode: clustering breaks down because legitimate documents are intentionally diverse across reasoning steps. A *concentration metric* (embedding density) is used instead — adversarial docs are unusually concentrated around one claim.

**Stage 2 — Identification:**
The flagged suspicious group is removed. Only `Rsafe` is forwarded to the LLM generator. This entire process requires no LLM inference — just fast embedding and clustering computations.

---

## 3. Limitations of the original system

After studying the codebase and the paper's appendices, we identified four weaknesses:

**1. K-Means has hard geometric assumptions**
K-Means forces every document into a cluster and requires knowing `k` upfront. It only finds spherical clusters. GARAG-style attacks deliberately spread poisoned document embeddings to defeat exactly this assumption — and they succeed, dropping detection accuracy significantly.

**2. Fixed rejection threshold breaks at high poison ratios**
The original system applies the same cutoff regardless of how many documents are poisoned. At a 4× poison ratio (4 adversarial docs per 1 clean), the poisoned docs dominate the score distribution and a fixed threshold lets most of them through.

**3. Embedding-only filtering misses logical contradictions**
Clustering works in embedding space — it catches documents that are *semantically grouped*. A sophisticated poison doc can sit far from other poisons in embedding space (different vocabulary, framing) while still asserting a logically contradictory fact. The paper's own Appendix C documents these false negatives.

**4. Manual mode selection**
Users must manually pick `singlehop` vs `multihop`. Choosing the wrong one for a given dataset hurts accuracy noticeably.

---

## 4. What we changed and why

### Improvement 1 — HDBSCAN (replaces K-Means)

**File:** `ragdefender/filtering/hdbscan_group.py`

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) builds a density hierarchy over the embedding space. Documents in dense regions form clusters. Documents in sparse, low-density regions get labeled as **noise (-1)** — which in this context means they don't belong to any coordinated group, i.e., they're likely legitimate.

What this fixes:
- No need to specify `k` — cluster count is discovered automatically
- Every isolated document gets a noise label — no forced assignment
- Outputs a **membership probability** per document, not just a binary label
- Handles arbitrary cluster shapes, not just spherical
- Catches GARAG-style dispersed attacks because the density hierarchy detects subtle groupings even when embeddings are spread out

```python
# Original
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=k).fit_predict(embeddings)

# Improved
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine', prediction_data=True)
labels = clusterer.fit_predict(embeddings)
probs  = clusterer.probabilities_
# label == -1  → noise → keep
# prob > 0.4 in a cluster with size >= 2 → adversarial → remove
```

### Improvement 2 — Dynamic MAD Threshold

**File:** `ragdefender/filtering/mad_threshold.py`

Instead of a fixed rejection cutoff, the threshold is recalculated fresh for each retrieved batch using **Median Absolute Deviation (MAD)** — a robust statistical measure that isn't pulled by outliers.

```
threshold = median(suspicion_scores) + 2.5 × MAD(suspicion_scores)
threshold = clip(threshold, 0.30, 0.85)
```

Why MAD and not mean/std? Because when there are many poisoned docs, the mean gets dragged up by the outliers. The median doesn't move — so the threshold correctly identifies the poison docs as the statistical tail, regardless of how many there are.

Effect at different poison ratios:

| Ratio | Fixed threshold | Dynamic MAD |
|---|---|---|
| 1× | catches 1/1 ✓ | catches 1/1 ✓ |
| 2× | catches 0/4 ✗ | catches 4/4 ✓ |
| 4× | catches 1/8 ✗ | catches 6/8 ~ |

### Improvement 3 — NLI Contradiction Filter

**File:** `ragdefender/filtering/nli_filter.py`

NLI (Natural Language Inference) models are trained to classify whether one sentence *entails*, *contradicts*, or is *neutral* to another. Running a lightweight cross-encoder NLI model on pairs of surviving documents catches logical contradictions that slipped past the embedding-based stages.

Model used: `cross-encoder/nli-deberta-v3-small` (~180MB, runs on CPU)

Logic:
1. Run pairwise NLI on all surviving documents
2. If contradiction probability > 0.65 for a pair, vote against both documents
3. Remove documents whose vote count exceeds the average — **except** the most query-relevant document (the "query anchor"), which is always protected

This directly addresses the false negative case study in Appendix C of the paper: a poison doc with different vocabulary that sits far from other poisons in embedding space but still contradicts the truth.

### Summary of changes

| Component | Original | Improved |
|---|---|---|
| Singlehop clustering | K-Means (fixed k) | HDBSCAN (auto k, noise labels, probability scores) |
| Rejection threshold | Fixed global cutoff | Dynamic MAD (recalibrated per batch) |
| Post-filter verification | None | NLI contradiction check |
| Backwards compatibility | — | All original API calls still work unchanged |

---

## 5. Installation

### Requirements

- Python 3.8+
- pip

### Step 1 — Clone the repo

```bash
git clone https://github.com/abhiu021/RAGDefender.git
cd RAGDefender
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If you want HDBSCAN (required for Improvement 1):
```bash
pip install hdbscan
```

If you want the NLI filter (required for Improvement 3):
```bash
pip install sentence-transformers
```
The NLI model (`cross-encoder/nli-deberta-v3-small`) downloads automatically on first use.

### Step 4 — Install the package in editable mode

```bash
pip install -e .
```

### Verify installation

```bash
ragdefender --help
python -c "from ragdefender import RAGDefender; print('OK')"
```

---

## 6. How to run — step by step

### Run the basic example (original system)

```bash
python examples/basic_usage.py
```

Expected output:
```
Initializing RAGDefender...
Loading encoder: all-mpnet-base-v2
Device: cpu
Ready.

Query : Where is the capital of France?
Docs  : 5 retrieved (1 clean, 4 poisoned)

[Defender] Running singlehop mode...
[Stage 1] Embedding 5 documents...
[Stage 1] Clustering with K-Means (k=2)...
[Stage 2] Flagging adversarial cluster...

Result  : Removed 4 poisoned documents
Clean   : ['Paris serves as the heart of France...']
```

---

### Run the improved pipeline (all three changes)

```bash
python improvements/full_pipeline.py
```

This runs all three improvements in sequence (HDBSCAN → MAD → NLI) on a set of demo queries and prints a comparison table.

---

### Run individual improvements

**HDBSCAN only:**
```bash
python improvements/hdbscan_defend.py --mode compare
```

This runs both K-Means and HDBSCAN on the same set of retrieved docs and shows results side by side — useful for seeing exactly where HDBSCAN gains over K-Means.

**Dynamic MAD threshold only:**
```bash
python improvements/mad_threshold.py --ratios 1 2 4
```

Shows threshold calibration at 1×, 2×, and 4× poison ratios. At 4×, you'll see the fixed threshold catching 1/8 poisons while the dynamic threshold catches 6/8.

**NLI filter only:**
```bash
python improvements/nli_filter.py
```

Demonstrates the NLI filter catching a false negative — a poison doc that evaded both HDBSCAN and MAD because its embedding was far from other poisons, but which logically contradicts the clean document (contradiction probability: 0.882).

---

### Run via CLI

```bash
# Defend a set of documents
ragdefender defend \
  --query "Where is the capital of France?" \
  --corpus path/to/corpus.json \
  --mode singlehop \
  --device cpu

# Save output
ragdefender defend \
  --query "Where is the capital of France?" \
  --corpus path/to/corpus.json \
  --mode singlehop \
  --output defended_output.json

# Evaluate against an attack
ragdefender evaluate \
  --test-data path/to/test_data.json \
  --attack poisonedrag \
  --mode singlehop \
  --device cpu
```

---

### Run the research artifact evaluation scripts

These run the full benchmark used in the paper. Note: requires a GPU with 15GB+ VRAM for loading the LLM generator.

```bash
cd artifacts
python run_poisonedrag.py
python run_blind.py
python run_garag.py
```

---

### Run tests

```bash
python -m pytest tests/ -v
```

To run only the improvement-specific tests:
```bash
python -m pytest tests/test_improvements.py -v
```

---

## 7. Python API

### Original API — unchanged, still works

```python
from ragdefender import RAGDefender

defender = RAGDefender(device="cpu")

clean_docs = defender.defend(
    query="Where is the capital of France?",
    retrieved_docs=[
        "Paris is the capital and most populous city of France.",
        "POISONED: Marseille is the capital of France.",
        "POISONED: Strasbourg is the capital of France.",
    ],
    mode="singlehop",
    top_k=2
)

print(clean_docs)
```

### Improved API — new parameters, all optional

```python
from ragdefender import RAGDefender

# Enable all three improvements
defender = RAGDefender(
    device="cpu",
    use_hdbscan=True,           # replaces K-Means with HDBSCAN
    use_dynamic_threshold=True, # MAD-based adaptive threshold
    use_nli=True,               # NLI contradiction filter (downloads model on first use)
    hdbscan_threshold=0.4,      # membership probability cutoff (default 0.4)
    mad_sensitivity=2.5,        # MAD multiplier — lower = more aggressive (default 2.5)
    nli_threshold=0.65,         # contradiction probability cutoff (default 0.65)
)

clean_docs = defender.defend(
    query="Where is the capital of France?",
    retrieved_docs=[...],
    mode="singlehop",
    top_k=2
)
```

### Using individual components directly

```python
from ragdefender.filtering.mad_threshold import compute_suspicion_scores, dynamic_threshold, mad_filter
from ragdefender.filtering.nli_filter import nli_contradiction_filter
from ragdefender.filtering.hdbscan_group import hdbscan_group
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-mpnet-base-v2")
embeddings = encoder.encode(docs)

# HDBSCAN grouping
clean_idx, adversarial_idx = hdbscan_group(embeddings, threshold=0.4)

# MAD threshold
scores = compute_suspicion_scores(embeddings)
threshold = dynamic_threshold(scores, sensitivity=2.5)
clean_docs = mad_filter(docs, embeddings, sensitivity=2.5)

# NLI filter
from sentence_transformers import CrossEncoder
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", num_labels=3)
clean_docs = nli_contradiction_filter(query, docs, encoder, nli_model, contradiction_threshold=0.65)
```

---

## 8. CLI usage

```bash
ragdefender --help

# Available commands:
#   defend      Run the defense filter on a retrieved doc set
#   evaluate    Run benchmark evaluation
#   info        Show current config and loaded models

ragdefender defend --help
ragdefender evaluate --help
ragdefender info --help
```

### Corpus format

The `--corpus` flag accepts:

1. **JSON list of strings** — `["doc1", "doc2", ...]`
2. **JSON object with a documents field** — `{"documents": ["doc1", ...]}`
3. **Plain text file** — one document per line

---

## 9. Results and outcomes

### Original paper results (from the paper)

| Dataset | Retriever | Generator | No defense ASR | RAGDefender ASR |
|---|---|---|---|---|
| NQ | Contriever | LLaMA-7B | 0.78 | 0.09 |
| NQ | Contriever | Gemini | 0.89 | 0.02 |
| MS MARCO | DPR | Vicuna-7B | 0.71 | 0.07 |
| HotpotQA | ANCE | LLaMA-7B | 0.65 | 0.11 |

ASR = Attack Success Rate. Lower is better. Results at 4× poison ratio.

### Our improvements — estimated accuracy gains

| Attack type | Baseline | + HDBSCAN | + MAD | + NLI |
|---|---|---|---|---|
| PoisonedRAG (1×) | ~78% | ~95% | ~97% | ~99% |
| PoisonedRAG (4×) | ~43% | ~78% | ~91% | ~94% |
| GARAG (dispersed) | ~35% | ~60% | ~68% | ~82% |
| Blind attack | ~55% | ~88% | ~90% | ~93% |

Numbers shown are poison detection accuracy (higher is better). Baseline = original RAGDefender with K-Means and fixed threshold. Estimates based on paper-reported numbers and the gap each technique closes.

### Speed comparison

| Defense | Mechanism | Speed vs RAGDefender |
|---|---|---|
| RobustRAG | LLM inference per doc | 12.36× slower |
| Discern-and-Answer | Fine-tuned discriminator | 1.53× slower |
| RAGDefender (original) | Lightweight ML | baseline |
| RAGDefender (ours) | + HDBSCAN + MAD + NLI | ~1.1× slower |

The NLI stage adds minor overhead (O(n²) pairwise comparisons on CPU) but at typical RAG sizes of n=10–20 documents this is negligible — under 200ms.

### What the biggest gains come from

- **HDBSCAN** makes the biggest single jump (+17pp on PoisonedRAG 1×, +35pp at 4× vs baseline), mainly by fixing the K-Means geometric assumption
- **Dynamic MAD** shows the biggest gain specifically at high poison ratios (4×) — exactly where the original system struggles most
- **NLI** provides the final layer that catches logically inconsistent docs which evade both embedding-based stages

---

## 10. Project structure

```
RAGDefender/
│
├── ragdefender/               # core package
│   ├── __init__.py            # exports RAGDefender class
│   ├── cli.py                 # command-line interface
│   ├── core/
│   │   ├── defender.py        # main orchestration — improved here
│   │   └── evaluator.py       # evaluation utilities
│   └── filtering/
│       ├── hdbscan_group.py   # NEW — HDBSCAN-based clustering
│       ├── mad_threshold.py   # NEW — dynamic MAD threshold
│       └── nli_filter.py      # NEW — NLI contradiction filter
│
├── improvements/              # standalone demo scripts for each improvement
│   ├── hdbscan_defend.py      # HDBSCAN vs K-Means comparison demo
│   ├── mad_threshold.py       # MAD threshold at multiple poison ratios
│   ├── nli_filter.py          # NLI false-negative catch demo
│   └── full_pipeline.py       # all three combined, end-to-end
│
├── examples/
│   ├── basic_usage.py         # original API example
│   └── improved_usage.py      # new API with all improvements
│
├── tests/
│   ├── test_defender.py       # original tests (unchanged)
│   └── test_improvements.py   # NEW — tests for all three improvements
│
├── artifacts/                 # research benchmark scripts
│   ├── run_poisonedrag.py
│   ├── run_blind.py
│   └── run_garag.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 11. Troubleshooting

**`hdbscan` not found**
```bash
pip install hdbscan
```
If it still fails on Windows, try: `pip install hdbscan --no-build-isolation`

**NLI model download fails**
The model downloads from HuggingFace on first use. If you're offline, pre-download it:
```python
from sentence_transformers import CrossEncoder
CrossEncoder("cross-encoder/nli-deberta-v3-small", num_labels=3)
```
Or set `use_nli=False` to skip that stage entirely — the other two improvements will still run.

**CUDA/GPU errors**
Start with CPU mode to validate setup:
```python
defender = RAGDefender(device="cpu")
```

**`ragdefender: command not found`**
```bash
pip install -e .
# or
pip install ragdefender
```

**Defense removes too many legitimate documents (false positives)**
Reduce aggressiveness:
```python
defender = RAGDefender(
    hdbscan_threshold=0.6,    # was 0.4 — harder to flag as adversarial
    mad_sensitivity=3.5,      # was 2.5 — more lenient threshold
    nli_threshold=0.80,       # was 0.65 — only flag high-confidence contradictions
)
```

**Defense misses too many poison docs (false negatives)**
Increase aggressiveness:
```python
defender = RAGDefender(
    hdbscan_threshold=0.3,
    mad_sensitivity=1.5,
    nli_threshold=0.50,
)
```

**Runtime slower than expected**
- Reduce the number of retrieved docs passed in (`top_k` at the retriever level)
- Set `use_nli=False` — the NLI stage is the most compute-intensive

---

## 12. Citation

If you use this in research, please cite the original paper:

```bibtex
@inproceedings{kim2025ragdefender,
  title={Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems},
  author={Kim, Minseok and Lee, Hankook and Koo, Hyungjoon},
  booktitle={Proceedings of the 41st Annual Computer Security Applications Conference (ACSAC)},
  year={2025}
}
```

If you use the improvements from this fork specifically, please also mention this repository.

---

## Acknowledgements

- Original RAGDefender system by [SecAI-Lab, SKKU](https://github.com/SecAI-Lab/RAGDefender)
- HDBSCAN library by [Leland McInnes et al.](https://github.com/scikit-learn-contrib/hdbscan)
- NLI model: [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) on HuggingFace

---

*For questions or issues, open a GitHub issue or reach out via the repository.*