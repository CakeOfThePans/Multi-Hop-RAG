import os
from dotenv import load_dotenv
from datasets import load_dataset
import argparse

from retrievers.faiss_retriever import FaissRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever

from models.multi_hop import MultiHopQA
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--retrieval_mode", type=str, default="faiss")  # faiss, bm25, hybrid
    parser.add_argument("--dataset_name", type=str, default="hotpot")   # hotpot, musique, 2wiki
    parser.add_argument("--k_retrieve", type=int, default=5)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--index_dir", type=str, default="vector_stores")

    args = parser.parse_args()
    retrieval_mode = args.retrieval_mode
    dataset_name = args.dataset_name
    k_retrieve = args.k_retrieve
    n_eval = args.n_eval
    index_dir = args.index_dir

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

    model = MultiHopQA(
        retriever=retriever,
        chat_model="gpt-4o-mini",
        temperature=0.0,
        max_hops=3,
        max_docs_per_hop=3,
    )

    ds_val = load_validation_set(dataset_name)

    def predict_fn(question: str, k: int):
        return model.predict(question, k=k)

    metrics = evaluate_qa_system(ds_val, predict_fn, n=n_eval, k=k_retrieve)

    print("\n=====================================")
    print(f"  DATASET: {dataset_name}")
    print(f"  RETRIEVER: {retrieval_mode}")
    print("  RESULTS:", metrics)
    print("=====================================\n")

if __name__ == "__main__":
    main()