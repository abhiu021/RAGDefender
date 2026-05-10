# RAGDefender

[![PyPI version](https://badge.fury.io/py/ragdefender.svg)](https://badge.fury.io/py/ragdefender)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

RAGDefender is a defense library for Retrieval-Augmented Generation (RAG) systems. It is designed to detect and remove poisoned retrieval results before those documents are sent to an LLM.

It targets knowledge-corruption attacks such as:
- **PoisonedRAG**
- **Blind**
- **GARAG**

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Installation](#installation)
5. [How to run the code](#how-to-run-the-code)
6. [Python API usage](#python-api-usage)
7. [CLI usage](#cli-usage)
8. [Evaluation workflow](#evaluation-workflow)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

---

## What this project does

RAGDefender adds a filtering/selection stage between retrieval and generation:

1. Your retriever returns candidate passages.
2. RAGDefender scores and filters likely-poisoned passages.
3. Your generator receives cleaner context.

The goal is to reduce attack success rate while preserving useful evidence.

---

## How it works

RAGDefender exposes a single high-level entry point (`RAGDefender`) and supports two operation modes:

- **singlehop**: for simpler factual queries (commonly NQ/MSMARCO style)
- **multihop**: for reasoning queries requiring multiple supporting documents (commonly HotpotQA style)

You pass:
- `query` (user question)
- `retrieved_docs` (list of strings)
- `mode` (`singlehop` or `multihop`)
- optional `top_k`

RAGDefender returns a cleaned document list.

---

## Project structure

```text
ragdefender/                  # pip package source
  cli.py                      # command-line entry point
  core/                       # defender + evaluator core logic
  filtering/                  # filtering modules
  attacks/, datasets/, models/, defenses/

examples/                     # small usage examples
artifacts/                    # research/evaluation scripts and data artifacts
QUICKSTART.md                 # quick guide
README_PYPI.md                # package-focused readme
requirements.txt              # dependencies for local/dev usage
pyproject.toml, setup.py      # package metadata/build config
```

---

## Installation

### Option A: Install from PyPI (recommended for normal use)

```bash
pip install ragdefender
```

GPU extras:

```bash
pip install ragdefender[cuda]
```

### Option B: Install from source (recommended for development/research)

```bash
git clone https://github.com/SecAI-Lab/RAGDefender.git
cd RAGDefender
pip install -e .
```

If you need all local dependencies listed by the repository:

```bash
pip install -r requirements.txt
```

---

## How to run the code

This section gives concrete run commands.

### 1) Run a Python example

```bash
python examples/basic_usage.py
```

### 2) Run package CLI help (sanity check)

```bash
ragdefender --help
```

### 3) Defend a query with CLI

Prepare a corpus file:
- JSON list of strings, or
- JSON object with `documents`, or
- plain text (one document per line)

Then run:

```bash
ragdefender defend \
  --query "Where is the capital of France?" \
  --corpus path/to/corpus.json \
  --mode multihop \
  --device cpu
```

Optional output file:

```bash
ragdefender defend \
  --query "Where is the capital of France?" \
  --corpus path/to/corpus.json \
  --mode multihop \
  --output defended_output.json
```

### 4) Run evaluation on labeled test data

```bash
ragdefender evaluate \
  --test-data path/to/test_data.json \
  --attack poisonedrag \
  --mode multihop \
  --device cpu
```

### 5) Run research artifact scripts

```bash
cd artifacts
python run_poisonedrag.py
python run_blind.py
python run_garag.py
```

---

## Python API usage

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
    top_k=2             # optional
)

print(clean_docs)
```

---

## CLI usage

Available subcommands:
- `ragdefender defend`
- `ragdefender evaluate`
- `ragdefender info`

Get help:

```bash
ragdefender defend --help
ragdefender evaluate --help
ragdefender info --help
```

---

## Evaluation workflow

Typical loop:
1. Build or load attacked retrieval corpora.
2. Run defense (`defend` API/CLI).
3. Compare metrics (precision/recall/F1 and downstream QA impact).
4. Repeat by attack type (`poisonedrag`, `blind`, `garag`) and query mode.

For quick examples, see:
- `QUICKSTART.md`
- `examples/basic_usage.py`

---

## Troubleshooting

- **Command not found: `ragdefender`**
  - Install package first: `pip install ragdefender` or `pip install -e .`

- **CUDA issues / no GPU**
  - Use CPU mode: `--device cpu` or `RAGDefender(device="cpu")`

- **Corpus format error in CLI**
  - Ensure JSON is either a list of strings or an object with a `documents` key.

- **Slow runtime**
  - Reduce input document count and/or use `top_k`.

---

## Citation

If you use this project in research, cite the paper:

```bibtex
@inproceedings{kim2025ragdefender,
  title={Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems},
  author={Kim, Minseok and Lee, Hankook and Koo, Hyungjoon},
  booktitle={Proceedings of the 41st Annual Computer Security Applications Conference (ACSAC)},
  year={2025}
}
```
