# Multi-Hop Retrieval-Augmented Generation (RAG) Benchmarking

This repository provides a modular framework for evaluating single-hop and multi-hop Retrieval-Augmented Generation (RAG) systems across multiple multi-hop QA datasets.

It supports:

- FAISS dense retrieval
- BM25 sparse retrieval
- Hybrid retrieval (dense + sparse via RRF)
- Cross-encoder reranking
- Single-hop RAG
- Multi-hop RAG with question decomposition
- EM, F1, and LLM semantic evaluation
- Datasets: HotpotQA, MuSiQue, and 2WikiMultiHopQA

---

## Repository Structure

```
.
├── retrievers/
│   ├── faiss_retriever.py          # Dense vector search (FAISS)
│   ├── bm25_retriever.py           # Sparse BM25 retrieval
│   ├── hybrid_retriever.py         # Dense + Sparse fusion via RRF
│   └── reranker.py                 # Cross-encoder reranking
│
├── models/
│   ├── single_hop.py               # Standard single-hop RAG pipeline
│   └── multi_hop.py                # Multi-hop RAG + decomposition pipeline
│
├── utils/
│   ├── chunking.py                 # Corpus building + chunking
│   ├── prompts.py                  # All LLM prompts (single + multi-hop)
│   ├── jsonl_utils.py              # Utilities for JSONL files (BM25)
│   └── eval.py                     # EM/F1/LLM evaluation functions
│
├── run_singlehop.py                # CLI runner for single-hop evaluation
├── run_multihop.py                 # CLI runner for multi-hop evaluation
├── reasoning_trace_comparison.py   # Compare single vs multi-hop on one question
├── build_vector_store.py           # Build FAISS index + JSONL corpus
│
├── corpora/                        # Auto-generated after building vector store
├── vector_stores/                  # Auto-generated FAISS directories
│
├── requirements.txt
└── README.md
```

---

## Required Dataset Downloads

### 1. HotpotQA & MuSiQue
These are automatically downloaded from HuggingFace — no action needed.

---

### 2. 2WikiMultiHopQA (must be downloaded manually)

The original 2Wiki dataset is no longer hosted on HuggingFace.  
You must download it manually:

Download link:  
https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?e=2&file_subpath=%2Fdata&rlkey=ira57daau8lxfj022xvk1irju

Inside the ZIP you will find:

```
train.json
dev.json
test.json
```

Place them in a folder named exactly: `2wiki/`

So your project root should look like:

```
.
├── 2wiki/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── retrievers/
├── models/
├── utils/
...
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Create a `.env` file in the repo root:

```ini
OPENAI_API_KEY=your_key_here
```

---

## Step 1 — Build Vector Stores (FAISS + JSONL)

This step:
- Loads datasets
- Cleans + deduplicates paragraphs
- Chunks all documents
- Saves JSONL corpus for BM25
- Builds FAISS index for dense retrieval

Run:

```bash
python build_vector_store.py
```

You will be prompted:

```
Select dataset to build vector store:
[1] HotpotQA
[2] MuSiQue
[3] 2WikiMultiHopQA
```

Afterwards, the repo will contain:

```
vector_stores/
    faiss_hotpot/
    faiss_musique/
    faiss_2wiki/

corpora/
    hotpot.jsonl
    musique.jsonl
    2wiki.jsonl
```

You only need to run this once per dataset unless you change embeddings or chunking size.

---

## Step 2 — Run Single-Hop RAG

```bash
python run_singlehop.py
```

### Arguments

| Flag | Meaning | Default |
|------|---------|---------|
| `--retrieval_mode` | `faiss` / `bm25` / `hybrid` | `faiss` |
| `--dataset_name` | `hotpot` / `musique` / `2wiki` | `hotpot` |
| `--k_retrieve` | Number of retrieved docs | `5` |
| `--n_eval` | Number of validation samples | `100` |
| `--index_dir` | Path to FAISS indexes | `vector_stores` |
| `--rerank` | Enable cross-encoder reranking | `false` |

### Example

```bash
python run_singlehop.py --retrieval_mode hybrid --dataset_name 2wiki --k_retrieve 5 --n_eval 50 --rerank
```

---

## Step 3 — Run Multi-Hop RAG

```bash
python run_multihop.py
```

Uses:
- LLM-based question decomposition
- Per-hop retrieval + answering
- Final synthesis model

### Arguments

| Flag | Meaning | Default |
|------|---------|---------|
| `--retrieval_mode` | `faiss` / `bm25` / `hybrid` | `faiss` |
| `--dataset_name` | `hotpot` / `musique` / `2wiki` | `hotpot` |
| `--k_retrieve` | Number of retrieved docs | `5` |
| `--n_eval` | Number of validation samples | `100` |
| `--index_dir` | Path to FAISS indexes | `vector_stores` |
| `--rerank` | Enable cross-encoder reranking | `false` |

### Example

```bash
python run_multihop.py --retrieval_mode faiss --dataset_name hotpot --k_retrieve 5 --n_eval 10 --rerank
```

---

## Step 4 — Compare Single-Hop vs Multi-Hop on One Question

```bash
python reasoning_trace_comparison.py
```

### Arguments

| Flag | Meaning | Default |
|------|---------|---------|
| `--retrieval_mode` | `faiss` / `bm25` / `hybrid` | `faiss` |
| `--dataset_name` | `hotpot` / `musique` / `2wiki` | `hotpot` |
| `--k_retrieve` | Number of retrieved docs | `5` |
| `--question_idx` | Question index from the validation set | `4` |
| `--index_dir` | Path to FAISS indexes | `vector_stores` |
| `--rerank` | Enable cross-encoder reranking | `false` |

### Example

```bash
python reasoning_trace_comparison.py --retrieval_mode faiss --dataset_name hotpot --k_retrieve 5 --question_idx 4 --rerank
```

This prints:
- Original question and ground truth answer
- Reasoning trace for single-hop, including:
    - Retrived passages
    - Final model answer
    - Evaluation metrics
- Reasoning trace for multi-hop, including:
    - Sub-questions generated
    - Retrieved passages
    - Intermediate answers
    - Final model answer
    - Evaluation Metrics

---

## Retrieval Modes

| Mode | Description |
|------|-------------|
| `faiss` | Dense retrieval using BGE embeddings |
| `bm25` | Classic sparse retrieval |
| `hybrid` | Reciprocal Rank Fusion over dense + sparse |

Hybrid almost always improves recall.

---

## Single-Hop Architecture

`single_hop.py` performs:

1. Retrieve top-k contexts
2. Construct a single-pass prompt
3. Produce final answer
4. Return answer + evaluation metadata

This baseline measures the power of retrieval alone.

---

## Multi-Hop Architecture

`multi_hop.py` performs:

1. LLM-based question decomposition into sub-questions
2. Retrieval per hop
3. Intermediate answering
4. Final answer synthesis

The decomposition relies on structured outputs (Pydantic), requiring models that support `response_format=json_schema`.

---

## Vector Store Persistence

- FAISS retriever: Stored on disk (e.g., `vector_stores/faiss_hotpot/`) to avoid re-embedding hundreds of thousands of chunks each run.
- BM25 retriever: Loads corpus from `.jsonl` file inside `corpora/` at runtime to build the BM25 index.

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| EM (Exact Match) | Strict correctness — prediction must match ground truth exactly |
| F1 | Token-level overlap balancing precision and recall |
| LLM Eval | Semantic scoring using an LLM judge |
