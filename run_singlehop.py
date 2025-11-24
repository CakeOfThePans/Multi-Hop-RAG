import os
from dotenv import load_dotenv
from datasets import load_dataset

from retrievers.faiss_retriever import FaissRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever

from models.single_hop import SingleHopQA
from utils.eval import evaluate_qa_system


def load_validation_set(name: str):
    """Load the validation split for each dataset."""
    if name == "hotpot":
        return load_dataset("hotpot_qa", "fullwiki", split="validation")
    elif name == "musique":
        return load_dataset("dgslibisey/MuSiQue", split="validation")
    elif name == "2wiki":
        return load_dataset("framolfese/2WikiMultihopQA", split="validation")
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    load_dotenv()

    retrieval_mode = os.getenv("RETRIEVAL_MODE", "faiss")     # faiss, bm25, hybrid
    dataset_name = os.getenv("DATASET_NAME", "hotpot")      # hotpot, musique, 2wiki
    k_retrieve = int(os.getenv("K_RETRIEVE", 5))
    n_eval = int(os.getenv("N_EVAL", 100))
    index_dir = os.getenv("INDEX_DIR", "vector_stores")   # where FAISS indices are stored


    if retrieval_mode == "faiss":
        retriever = FaissRetriever(dataset_name, index_dir=index_dir)

    elif retrieval_mode == "bm25":
        retriever = BM25Retriever(dataset_name)

    elif retrieval_mode == "hybrid":
        retriever = HybridRetriever(
            FaissRetriever(dataset_name, index_dir=index_dir),
            BM25Retriever(dataset_name),
            k_dense=k_retrieve,
            k_sparse=k_retrieve,
        )

    else:
        raise ValueError(f"Unknown RETRIEVAL_MODE: {retrieval_mode}")


    model = SingleHopQA(
        retriever=retriever,
        chat_model="gpt-4o-mini",
        temperature=0.0,
    )


    ds_val = load_validation_set(dataset_name)

    def predict_fn(question: str, k: int):
        return model.predict(question, k=k)

    metrics = evaluate_qa_system(ds_val, predict_fn, n=n_eval, k=k_retrieve)

    print("\n====================================")
    print(f"Dataset: {dataset_name}")
    print(f"Retriever: {retrieval_mode}")
    print("Results:", metrics)
    print("====================================\n")

if __name__ == "__main__":
    main()