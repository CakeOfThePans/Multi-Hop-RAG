import os
from typing import List
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ====================================================================
#                           FAISS RETRIEVER
# ====================================================================

class FaissRetriever:
    """
    Generic FAISS retriever for any dataset (hotpot_fullwiki, musique, 2wiki).
    Loads a prebuilt FAISS index created by build_vector_store.py.
    """

    def __init__(
        self,
        dataset_name: str,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        index_dir: str = "/vector_stores",
    ):
        """
        dataset_name: one of ["hotpot", "musique", "2wiki"]
        base_dir: directory where faiss_<dataset_name>/ folders live.
        """

        self.dataset_name = dataset_name.lower()
        self.index_path = os.path.join(index_dir, f"faiss_{self.dataset_name}")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"\n❌ FAISS index not found for dataset: '{self.dataset_name}'\n"
                f"Expected directory: {self.index_path}\n"
                f"Run build_vector_store.py first.\n"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FAISS Retriever] Using device: {device}")

        # Embedding model (must match what index was built with)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        print(f"[FAISS Retriever] Loading FAISS index from: {self.index_path}")
        self.vector_store = FAISS.load_local(
            self.index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        print(f"[FAISS Retriever] ✔ Loaded FAISS index successfully.\n")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)


# ====================================================================
#                           FAISS INDEX BUILDER
# ====================================================================

def build_faiss_index(
    texts,
    metas,
    output_dir: str,
    embedding_model_name: str = "BAAI/bge-large-en-v1.5",
):
    """
    Build a FAISS index from chunks.

    Acceptable chunk formats:
        1. (text, metadata)
        2. {"text": "...", "metadata": {...}}
        3. "raw text"
    """

    print(f"[FAISS Builder] Creating FAISS index at: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FAISS Builder] Using device: {device}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True
    )

    print(f"[FAISS Builder] Encoding… this may take a while.")

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metas,
    )

    os.makedirs(output_dir, exist_ok=True)
    vector_store.save_local(output_dir)

    print(f"[FAISS Builder] ✔ FAISS index saved to: {output_dir}\n")