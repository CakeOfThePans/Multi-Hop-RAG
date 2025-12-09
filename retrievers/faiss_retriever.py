import os
from typing import List
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class FaissRetriever:
    """
    Generic FAISS retriever for the datasets (hotpot, musique, 2wiki).
    """

    def __init__(self, dataset_name, embedding_model_name = "BAAI/bge-large-en-v1.5", index_dir = "/vector_stores"):
        self.dataset_name = dataset_name.lower()
        self.index_path = os.path.join(index_dir, f"faiss_{self.dataset_name}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

    def similarity_search(self, query, k = 5):
        return self.vector_store.similarity_search(query, k)

def build_faiss_index(texts, metas, output_dir, embedding_model_name = "BAAI/bge-large-en-v1.5"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True
    )

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metas,
    )

    os.makedirs(output_dir, exist_ok=True)
    vector_store.save_local(output_dir)