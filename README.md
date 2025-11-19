## Multi-Hop Retrieval-Augmented Generation (RAG) Benchmarking

This repository implements a modular, clean-architecture framework for evaluating single-hop and multi-hop RAG systems across multiple multi-hop QA datasets.  
It includes:

- FAISS vector retrieval
- BM25 sparse retrieval
- Hybrid retrieval (dense + sparse)
- Single-hop and multi-hop pipelines
- Evaluation (EM + F1)
- Support for multiple datasets (HotpotQA, MuSiQue, 2WikiMultiHopQA)

---

## Repository Structure

```
.
├── retrievers/
│   ├── faiss_retriever.py        # Dense vector search (FAISS)
│   ├── bm25_retriever.py         # Sparse BM25 retrieval
│   └── hybrid_retriever.py       # Dense + Sparse fusion via RRF
│
├── models/
│   ├── single_hop.py             # Standard single-hop RAG pipeline
│   └── multi_hop.py              # Multi-hop RAG + decomposition pipeline
│
├── utils/
│   ├── chunking.py               # Corpus building + chunking
│   ├── prompts.py                # All LLM prompts (single + multi-hop)
│   └── eval.py                   # EM/F1 evaluation functions
│
├── run_singlehop.py              # CLI runner for single-hop evaluation
├── run_multihop.py               # CLI runner for multi-hop evaluation
├── build_vector_store.py         # Dataset loading
│
├── requirements.txt
└── README.md
```

Helper utilities such as `scripts/build_vector_store.py` construct FAISS indices; running them will create directories like `faiss_hotpot_fullwiki/`, `faiss_musique/`, and `faiss_2wiki/`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

Create a `.env` file in the repo root:

```ini
OPENAI_API_KEY=your_key_here
```

---

## Step 1 — Build a Vector Store (one time per dataset)

FAISS vector stores are expensive to generate, so this step is only needed once per dataset.

Run:

```bash
python scripts/build_vector_store.py
```

You will be prompted:

```
Select dataset to build vector store:
[1] HotpotQA – fullwiki
[2] MuSiQue – full_v1
[3] 2WikiMultiHopQA – main

Enter choice:
```

This will generate:

```
faiss_hotpot_fullwiki/
faiss_musique/
faiss_2wiki/
```

After this, you do **not** need to rebuild unless you change embeddings or chunking logic.

---

## Step 2 — Run Single-Hop or Multi-Hop Inference

These scripts load an existing FAISS index and run predictions (no metrics).

Single-hop inference:

```bash
python models/single_hop.py --dataset hotpot
```

Multi-hop inference:

```bash
python models/multi_hop.py --dataset hotpot
```

Supported datasets:

- `hotpot`
- `musique`
- `2wiki`

Examples:

```bash
python models/multi_hop.py --dataset musique
python models/single_hop.py --dataset 2wiki
```

---

## Step 3 — Run Evaluation (EM / F1)

To compute metrics, use:

```bash
python utils/eval.py --model single --dataset hotpot --n 100
```

or:

```bash
python utils/eval.py --model multi --dataset hotpot --n 100
```

Arguments:

| Flag | Meaning |
|------|---------|
| `--model single/multi` | choose pipeline |
| `--dataset hotpot/musique/2wiki` | choose dataset |
| `--n 100` | number of validation samples to test |

Example full run:

```bash
python utils/eval.py --model multi --dataset musique --n 150
```

---

## Retrieval Modes

Select the retrieval backend via environment variable:

```
RETRIEVAL_MODE=faiss      # dense retrieval (default)
RETRIEVAL_MODE=bm25       # sparse retrieval
RETRIEVAL_MODE=hybrid     # RRF fusion (FAISS + BM25)
```

Example:

```bash
export RETRIEVAL_MODE=hybrid
```

Hybrid mode uses Reciprocal Rank Fusion to merge dense and sparse signals.

---

## Single-hop Pipeline Overview

`single_hop.py` performs:

- Retrieve top-k passages via the configured retriever
- Format retrieved context plus the question into the single-hop prompt
- Produce a concise answer (or `"Unknown"`)

This baseline establishes performance without intermediate decomposition.

---

## Multi-hop Pipeline Overview

`multi_hop.py` orchestrates:

- LLM-based question decomposition (structured output enforced via Pydantic)
- Iterative retrieval per sub-question, incorporating previous hop answers
- Hop-by-hop answer generation
- Final answer synthesis using all hop outputs and supporting passages

---

## Vector Store Persistence

- **FAISS retriever**: stored on disk (e.g., `faiss_hotpot_fullwiki/`) to avoid re-embedding hundreds of thousands of chunks each run.
- **BM25 retriever**: builds an in-memory inverted index; if memory is tight, restrict the dataset subset before indexing.

---

## Evaluation Utilities

`utils/eval.py` provides:

- `exact_match_score()`
- `f1_score()`
- `evaluate_qa_system()`