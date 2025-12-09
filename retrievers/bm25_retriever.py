import os
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LC_BM25
from utils.jsonl_utils import load_corpus_from_jsonl

class BM25Retriever:
    """
    Generic BM25 retriever for any dataset (hotpot, musique, 2wiki).
    Loads pre-chunked corpus stored as JSONL files created by build_vector_store.py.
    """

    def __init__(self, dataset_name, corpus_dir = "corpora", k = 5):
        self.dataset_name = dataset_name.lower()
        self.corpus_path = os.path.join(corpus_dir, f"{self.dataset_name}.jsonl")

        docs = load_corpus_from_jsonl(self.corpus_path)

        self.k = k
        self.retriever: LC_BM25 = LC_BM25.from_documents(docs)
        self.retriever.k = k

    def similarity_search(self, query, k = None):
        if k is None:
            k = self.k

        docs = self.retriever.invoke(query)
        return docs[:k]