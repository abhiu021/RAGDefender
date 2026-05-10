# RAGDefender

[![PyPI version](https://badge.fury.io/py/ragdefender.svg)](https://badge.fury.io/py/ragdefender)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

RAGDefender is a practical defense layer for Retrieval-Augmented Generation (RAG) pipelines.  
It helps you identify and filter suspicious or poisoned retrieval results **before** they are passed into your LLM.

In plain terms: if your retriever pulls in a mix of useful passages and adversarial junk, RAGDefender tries to keep the good evidence and drop the bad context.

---

## Why this project exists

RAG systems are often treated as if retrieval is trustworthy. In practice, retrieval corpora can be manipulated:

- an attacker injects misleading passages,
- those passages are ranked highly,
- your generator consumes them as if they were facts,
- and answer quality and safety degrade.

RAGDefender focuses on this exact gap between retrieval and generation.

It currently targets knowledge-corruption settings such as:
- **PoisonedRAG**
- **Blind**
- **GARAG**

---

## What RAGDefender does (at a glance)

RAGDefender inserts a filtering stage into your RAG flow:

1. Retriever returns candidate passages.
2. RAGDefender scores and filters likely-poisoned candidates.
3. Generator receives a cleaner context set.

The objective is to lower attack success while keeping relevant evidence for downstream QA.

---

## Table of Contents

1. [Core concepts](#core-concepts)
2. [How the package is organized](#how-the-package-is-organized)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Python API](#python-api)
6. [CLI usage](#cli-usage)
7. [Input/output expectations](#inputoutput-expectations)
8. [Evaluation workflow](#evaluation-workflow)
9. [Reproducibility and artifacts](#reproducibility-and-artifacts)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

---

## Core concepts

### Operation modes

RAGDefender exposes a high-level entry point (`RAGDefender`) and supports two modes:

- **singlehop**: suited for simpler factoid-style questions (common in NQ/MSMARCO-like settings)
- **multihop**: suited for questions that require combining evidence from multiple documents (common in HotpotQA-like settings)

### Main inputs

You provide:

- `query`: the user question string
- `retrieved_docs`: list of retrieved document/passages (strings)
- `mode`: `singlehop` or `multihop`
- `top_k` (optional): number of defended documents you want returned

### Main output

- A cleaned list of documents/passages for your generator to consume.

---

## How the package is organized

```text
ragdefender/
  __init__.py
  cli.py                     # command-line interface
  core/
    defender.py              # main defender orchestration
    evaluator.py             # evaluation utilities
  filtering/
    nli_filter.py            # NLI-based filtering
    mad_threshold.py         # thresholding helper
  attacks/, datasets/, defenses/, models/

examples/
  basic_usage.py             # minimal API example

artifacts/                   # research scripts + benchmark artifacts
QUICKSTART.md                # concise runbook
README_PYPI.md               # PyPI-focused README
```

If you're new, start with `examples/basic_usage.py`, then move to CLI and evaluation.

---

## Installation

### Option A — from PyPI (recommended for most users)

```bash
pip install ragdefender
```

Optional CUDA extra:

```bash
pip install ragdefender[cuda]
```

### Option B — from source (recommended for development/research)

```bash
git clone https://github.com/SecAI-Lab/RAGDefender.git
cd RAGDefender
pip install -e .
```

To install repository-pinned development dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1) Run the packaged example

```bash
python examples/basic_usage.py
```

### 2) Confirm CLI entrypoint works

```bash
ragdefender --help
```

### 3) Run a defense pass from CLI

```bash
ragdefender defend \
  --query "Where is the capital of France?" \
  --corpus path/to/corpus.json \
  --mode multihop \
  --device cpu
```

### 4) Save defended output

```bash
ragdefender defend \
  --query "Where is the capital of France?" \
  --corpus path/to/corpus.json \
  --mode multihop \
  --output defended_output.json
```

---

## Python API

```python
from ragdefender import RAGDefender

defender = RAGDefender(device="cpu")

query = "Where is the capital of France?"
retrieved_docs = [
    "Paris is the capital and most populous city of France.",
    "POISONED: Marseille is the capital of France.",
    "POISONED: Strasbourg is the capital of France.",
]

clean_docs = defender.defend(
    query=query,
    retrieved_docs=retrieved_docs,
    mode="multihop",   # or "singlehop"
    top_k=2              # optional
)

print(clean_docs)
```

### Integration tip

A common production pattern is:

- retriever returns top-N candidates,
- RAGDefender filters/re-ranks,
- generator receives top-K defended candidates.

This keeps your generator insulated from raw retrieval noise/adversarial content.

---

## CLI usage

Available commands:

- `ragdefender defend`
- `ragdefender evaluate`
- `ragdefender info`

Inspect command-specific options:

```bash
ragdefender defend --help
ragdefender evaluate --help
ragdefender info --help
```

---

## Input/output expectations

### Corpus formats accepted by CLI

For `--corpus`, use one of:

1. **JSON list of strings**
2. **JSON object** with a `documents` field
3. **Plain text file** (one document per line)

### Practical advice for better results

- Keep documents semantically focused (short passages often work better than very long mixed documents).
- Avoid passing near-duplicate chunks repeatedly.
- Tune `top_k` based on your generator context window and task complexity.

---

## Evaluation workflow

Typical loop used in experiments:

1. Build or load attacked retrieval corpora.
2. Run defense (`defend` via API or CLI).
3. Compare defense metrics (e.g., precision/recall/F1 where labels are available).
4. Measure downstream QA impact with and without defense.
5. Repeat across attack families (`poisonedrag`, `blind`, `garag`) and both query modes.

Example evaluation command:

```bash
ragdefender evaluate \
  --test-data path/to/test_data.json \
  --attack poisonedrag \
  --mode multihop \
  --device cpu
```

---

## Reproducibility and artifacts

The `artifacts/` directory includes scripts and data resources used for research-style experiments.

Typical runs:

```bash
cd artifacts
python run_poisonedrag.py
python run_blind.py
python run_garag.py
```

If you're reproducing results, review script assumptions (paths, model checkpoints, and hardware constraints) before launching long jobs.

---

## Troubleshooting

- **`ragdefender: command not found`**
  - Install the package first (`pip install ragdefender`) or use editable install (`pip install -e .`).

- **CUDA/GPU errors**
  - Start in CPU mode (`--device cpu`) to validate setup first.

- **Corpus format parsing errors**
  - Verify the corpus file is one of the supported formats listed above.

- **Runtime slower than expected**
  - Reduce candidate count from retrieval and/or lower `top_k`.

- **Unexpected filtering behavior**
  - Inspect the raw retrieved docs; heavily noisy/duplicated retrieval can degrade any downstream filter.

---

## Citation

If you use this project in research, please cite:

```bibtex
@inproceedings{kim2025ragdefender,
  title={Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems},
  author={Kim, Minseok and Lee, Hankook and Koo, Hyungjoon},
  booktitle={Proceedings of the 41st Annual Computer Security Applications Conference (ACSAC)},
  year={2025}
}
```
