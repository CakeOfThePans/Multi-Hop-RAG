import os
from typing import List
import json

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as BM25
from utils.jsonl_utils import load_corpus_from_jsonl

class BM25Retriever:
    """
    Generic BM25 retriever for any dataset (hotpot, musique, 2wiki).

    Loads pre-chunked corpus stored as JSONL files created by build_vector_store.py.

    Unlike FAISS, BM25 is fully in-memory and does not require a GPU,
    and indexing is nearly instant compared to dense embeddings.
    """

    def __init__(self, dataset_name: str, corpus_dir: str = "corpora", k: int = 5):
        self.dataset_name = dataset_name.lower()
        self.corpus_path = os.path.join(corpus_dir, f"{self.dataset_name}.jsonl")

        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(
                f"\nNo BM25 corpus found for '{self.dataset_name}'.\n"
                f"Expected: {self.corpus_path}\n"
                f"Run build_vector_store.py first.\n"
            )
        print(f"[BM25 Retriever] Loading corpus from {self.corpus_path}...")

        docs = load_corpus_from_jsonl(self.corpus_path)
        print(f"[BM25 Retriever] Loaded {len(docs)} documents.")

        self.retriever = BM25.from_documents(docs)
        self.k = k
        self.retriever.k = k

        print(f"[BM25 Retriever] Ready with k={k}.")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        docs = self.retriever.invoke(query)
        return docs[:k]