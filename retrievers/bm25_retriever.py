import os
from typing import List
import json

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

class BM25Retriever:
    """
    Generic BM25 retriever for any dataset (hotpot, musique, 2wiki).

    Loads pre-chunked corpus stored as JSONL files created by build_vector_store.py.

    Unlike FAISS, BM25 is fully in-memory and does not require a GPU,
    and indexing is nearly instant compared to dense embeddings.
    """

    def __init__(self, dataset_name: str, corpus_dir: str = "corpora", k: int = 15):
        self.dataset_name = dataset_name.lower()
        self.corpus_path = os.path.join(corpus_dir, f"{self.dataset_name}.jsonl")

        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(
                f"\nNo BM25 corpus found for '{self.dataset_name}'.\n"
                f"Expected: {self.corpus_path}\n"
                f"Run build_vector_store.py first.\n"
            )
        print(f"[BM25 Retriever] Loading corpus from {self.corpus_path}...")

        docs = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                docs.append(
                    Document(
                        page_content=entry["text"],
                        metadata=entry.get("metadata", {}),
                    )
                )
        print(f"[BM25 Retriever] Loaded {len(docs)} documents.")

        self.retriever = BM25Retriever.from_documents(docs)
        self.k = k
        self.retriever.k = k

        print(f"[BM25 Retriever] Ready with k={k}.")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        docs = self.retriever.get_relevant_documents(query)
        return docs[:k]