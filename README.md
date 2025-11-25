## Multi-Hop Retrieval-Augmented Generation (RAG) Benchmarking

This repository implements a modular, clean-architecture framework for evaluating single-hop and multi-hop RAG systems across multiple multi-hop QA datasets.  
It includes:

- FAISS vector retrieval
- BM25 sparse retrieval
- Hybrid retrieval (dense + sparse)
- Single-hop and multi-hop pipelines
- Evaluation (EM + F1 + LLM Eval)
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
│   ├── jsonl_utils.py            # Utilities functions to handle jsonl files for BM25
│   └── eval.py                   # EM/F1 evaluation functions
│
├── run_singlehop.py              # CLI runner for single-hop evaluation
├── run_multihop.py               # CLI runner for multi-hop evaluation
├── reasoning_trace_comparison.py # CLI runner for single question comparison using both models
├── build_vector_store.py         # Dataset loading
│
├── requirements.txt
└── README.md
```

Helper utilities such as `build_vector_store.py` construct FAISS indices; running them will create directories like `faiss_hotpot/`, `faiss_musique/`, and `faiss_2wiki/`, along with their respective corpus jsonl files.

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

## Step 1 — Build a Vector Store and Save Corpus (one time per dataset)

FAISS vector stores are expensive to generate, so this step is only needed once per dataset.

While building the vector store, the script will also save the cleaned + deduplicated corpus in JSONL format for later use with BM25.

Run:

```bash
python scripts/build_vector_store.py
```

You will be prompted:

```
Select dataset to build vector store:
[1] HotpotQA
[2] MuSiQue
[3] 2WikiMultiHopQA

Enter choice:
```

This will generate:

```
.
├── vector_stores/
│   ├── faiss_hotpot/
│   ├── faiss_musique/
│   └── faiss_2wiki.py/
│
├── corpora/
│   ├── hotpot.jsonl
│   ├── musique.jsonl
│   └── 2wiki.jsonl
```

- The vector_store/ folder contains the FAISS index and associated metadata for each dataset.

- The corpora/ folder contains one JSONL corpus file per dataset (each line = one paragraph with metadata), ready for BM25.

After this, you do **not** need to rebuild unless you change embeddings or chunking logic.

---

## Step 2 — Run Single-Hop or Multi-Hop Inference

These scripts load an existing FAISS index and run predictions with evaluation metrics

Single-hop inference:

```bash
python run_singlehop.py
```

Multi-hop inference:

```bash
python run_multihop.py
```

Arguments:

| Flag | Meaning | Default |
|------|---------|---------|
| `--retrieval_mode faiss/bm25/hybrid` | choose retrieval mode | faiss |
| `--dataset_name hotpot/musique/2wiki` | choose dataset | hotpot |
| `--k_retrieve (1-10)` | number of chunks to use as context | 5 |
| `--n (up to max size of validation set)` | number of validation samples to test | 100 |
| `--index_dir (path)` | path to faiss index folder (keep as default unless changed) | vector_stores |

Example full run:

```bash
python run_multihop.py --retrieval_mode faiss --dataset_name hotpot --k_retrieve 5 --n_eval 10
```

---

## Step 3 (Optional) — Compare single-hop and multi-hop on a single example

These scripts allow you to compare single-hop and multi-hop models on a single example in verbose mode

Comparison:

```bash
python reasoning_trace_comparison.py
```

Arguments:

| Flag | Meaning | Default |
|------|---------|---------|
| `--retrieval_mode faiss/bm25/hybrid` | choose retrieval mode | faiss |
| `--dataset_name hotpot/musique/2wiki` | choose dataset | hotpot |
| `--k_retrieve (1-10)` | number of chunks to use as context | 5 |
| `--question_idx (up to max size of validation set)` | question index to test | 4 |
| `--index_dir (path)` | path to faiss index folder (keep as default unless changed) | vector_stores |

Example full run:

```bash
python reasoning_trace_comparison.py --retrieval_mode faiss --dataset_name hotpot --k_retrieve 5 --question_idx 4
```

---

## Retrieval Modes

- **faiss** — dense retrieval (default)
- **bm25** — sparse retrieval
- **hybrid** — RRF fusion using FAISS + BM25

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

- **FAISS retriever**: stored on disk (e.g., `vector_stores/<dataset_name>`) to avoid re-embedding hundreds of thousands of chunks each run.
- **BM25 retriever**: stores the corpus as a `.jsonl` file inside `corpora/<dataset_name>.jsonl`, which is loaded at runtime to build the BM25 index.

---

## Evaluation Metrics

- **Exact Match (EM)** — measures whether the model's answer matches the ground-truth string exactly.
- **F1 Score** — token-level overlap metric that balances precision and recall.
- **LLM-based Evaluator** — uses a large language model to judge answer correctness based on semantics rather than surface form.