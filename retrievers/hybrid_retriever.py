from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

from retrievers.faiss_retriever import FaissHotpotRetriever
from retrievers.bm25_retriever import BM25HotpotRetriever


class HybridHotpotRetriever:
    """
    Hybrid retriever that fuses FAISS (dense) and BM25 (sparse) results
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, faiss_retriever: FaissHotpotRetriever, bm25_retriever: BM25HotpotRetriever, k_dense: int = 5, k_sparse: int = 5, k0: int = 60):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.k0 = k0

    def _rrf_fuse(self, dense_docs: List[Document], sparse_docs: List[Document], k: int) -> List[Document]:
        scores: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, Any]] = {}

        def add_docs(docs: List[Document], weight: float):
            for rank, doc in enumerate(docs):
                key = (
                    doc.page_content,
                    tuple(sorted(doc.metadata.items())),
                )
                if key not in scores:
                    scores[key] = {"doc": doc, "score": 0.0}
                scores[key]["score"] += weight * (1.0 / (self.k0 + rank + 1))

        add_docs(dense_docs, weight=1.0)
        add_docs(sparse_docs, weight=1.0)

        fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [entry["doc"] for entry in fused[:k]]

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        dense_docs = self.faiss_retriever.similarity_search(query, k=self.k_dense)
        sparse_docs = self.bm25_retriever.similarity_search(query, k=self.k_sparse)
        return self._rrf_fuse(dense_docs, sparse_docs, k=k)