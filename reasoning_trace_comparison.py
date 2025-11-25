import os
from dotenv import load_dotenv
from datasets import load_dataset

from retrievers.faiss_retriever import FaissRetriever
# from retrievers.bm25_retriever import BM25Retriever
# from retrievers.hybrid_retriever import HybridRetriever

from models.single_hop import SingleHopQA
from models.multi_hop import MultiHopQA
from utils.eval import f1_score, exact_match_score, llm_eval_score

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

    retrieval_mode = os.getenv("RETRIEVAL_MODE", "faiss") # faiss, bm25, hybrid
    dataset_name = os.getenv("DATASET_NAME", "hotpot") # hotpot, musique, 2wiki
    k_retrieve = int(os.getenv("K_RETRIEVE", 5))
    question_idx = int(os.getenv("QUESTION_IDX", 4)) # question 5 for the hotpotqa dataset is a good example
    index_dir = os.getenv("INDEX_DIR", "vector_stores")

    if retrieval_mode == "faiss":
        retriever = FaissRetriever(dataset_name, index_dir=index_dir)

    # elif retrieval_mode == "bm25":
    #     retriever = BM25Retriever(dataset_name)

    # elif retrieval_mode == "hybrid":
    #     retriever = HybridRetriever(
    #         FaissRetriever(dataset_name, index_dir=index_dir),
    #         BM25Retriever(dataset_name),
    #         k_dense=k_retrieve,
    #         k_sparse=k_retrieve,
    #     )

    else:
        raise ValueError(f"Unknown RETRIEVAL_MODE: {retrieval_mode}")

    singlehop_model = SingleHopQA(
        retriever=retriever,
        chat_model="gpt-4o-mini",
        temperature=0.0,
        verbose=True
    )

    multihop_model = MultiHopQA(
        retriever=retriever,
        chat_model="gpt-4o-mini",
        temperature=0.0,
        max_hops=3,
        max_docs_per_hop=3,
        verbose=True
    )

    ds_val = load_validation_set(dataset_name)
    if question_idx < 0 or question_idx >= len(ds_val):
        raise ValueError(f"Invalid index. Must be between 0 and {len(ds_val) - 1}")
    row = ds_val[question_idx]
    question = row["question"]
    answer = row["answer"]

    print("\n==================================================================")
    print(f"COMPARISON: Single-Hop vs Multi-Hop Reasoning ({dataset_name})")
    print(f"Question: \"{question}\"")
    print(f"Ground Truth:", answer)
    print("==================================================================\n")

    print("\n==================================================================")
    print("SINGLE-HOP RAG")
    single_hop_pred = singlehop_model.predict(question, k=k_retrieve)
    print("EM:", exact_match_score(single_hop_pred, answer))
    print("F1:", f1_score(single_hop_pred, answer))
    print("LLM Eval:", llm_eval_score(question, answer, single_hop_pred)["score"])
    print("==================================================================\n")

    print("\n==================================================================")
    print("MULTI-HOP RAG")
    multi_hop_pred = multihop_model.predict(question, k=k_retrieve)
    print("EM:", exact_match_score(multi_hop_pred, answer))
    print("F1:", f1_score(multi_hop_pred, answer))
    print("LLM Eval:", llm_eval_score(question, answer, multi_hop_pred)["score"])
    print("==================================================================\n")

if __name__ == "__main__":
    main()