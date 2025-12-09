from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document

class CrossEncoderReranker:
    """
    Cross-encoder reranker for query-document pairs.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    DUe to it being a small, fast, good for reranking top-k candidates.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query, docs, top_k = 5):
        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]

        with torch.no_grad():
            batch = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**batch).logits.squeeze(-1)
            scores = logits.detach().cpu().tolist()

        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        reranked = [d for _, d in scored_docs[:top_k]]
        return reranked


class RerankRetriever:
    """
    Thin wrapper around an existing retriever that applies a cross-encoder reranker on its similarity_search results.
    """
    def __init__(self, base_retriever, reranker, prefetch_k = 20):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.prefetch_k = prefetch_k

    def similarity_search(self, query, k = 5):
        pre_k = max(self.prefetch_k, k)
        base_docs = self.base_retriever.similarity_search(query, k=pre_k)
        return self.reranker.rerank(query, base_docs, top_k=k)