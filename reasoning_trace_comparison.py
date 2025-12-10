import argparse
import os
from datasets.load import json
from dotenv import load_dotenv
from datasets import load_dataset
from retrievers.faiss_retriever import FaissRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from models.single_hop import SingleHopQA
from models.multi_hop import MultiHopQA
from retrievers.reranker import CrossEncoderReranker, RerankRetriever
from utils.eval import f1_score, exact_match_score, llm_eval_score
from utils.phoenix_config import initialize_phoenix, get_phoenix_url, shutdown_phoenix

def load_validation_set(name):
    if name == "hotpot":
        return load_dataset("hotpot_qa", "fullwiki", split="validation")
    elif name == "musique":
        return load_dataset("dgslibisey/MuSiQue", split="validation")
    elif name == "2wiki":
        with open("2wiki/dev.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()

    parser.add_argument("--retrieval_mode", type=str, default="faiss")
    parser.add_argument("--dataset_name", type=str, default="hotpot")
    parser.add_argument("--k_retrieve", type=int, default=5)
    parser.add_argument("--question_idx", type=int, default=4)
    parser.add_argument("--index_dir", type=str, default="vector_stores")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking on top of the chosen retriever.",
    )
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="Enable Phoenix observability for tracing and error analysis.",
    )
    parser.add_argument(
        "--phoenix_port",
        type=int,
        default=6006,
        help="Port for Phoenix server (default: 6006)",
    )

    args = parser.parse_args()
    retrieval_mode = args.retrieval_mode
    dataset_name = args.dataset_name
    k_retrieve = args.k_retrieve
    question_idx = args.question_idx
    index_dir = args.index_dir
    use_rerank = args.rerank

    if retrieval_mode == "faiss":
        base_retriever = FaissRetriever(dataset_name, index_dir=index_dir)

    elif retrieval_mode == "bm25":
        base_retriever = BM25Retriever(dataset_name)

    elif retrieval_mode == "hybrid":
        base_retriever = HybridRetriever(
            FaissRetriever(dataset_name, index_dir=index_dir),
            BM25Retriever(dataset_name),
            k_dense=k_retrieve,
            k_sparse=k_retrieve,
        )

    if use_rerank:
        reranker = CrossEncoderReranker()
        base_retriever = RerankRetriever(
            base_retriever=base_retriever,
            reranker=reranker,
            prefetch_k=max(3 * k_retrieve, 20),
        )
        effective_retriever_name = f"{retrieval_mode}+rerank"
    else:
        effective_retriever_name = retrieval_mode

    # Initialize Phoenix if requested
    if args.phoenix:
        project_name = f"{dataset_name}_{effective_retriever_name}_comparison"
        tracer_provider = initialize_phoenix(
            project_name=project_name,
            port=args.phoenix_port,
            auto_instrument=True,
        )
        if tracer_provider:
            phoenix_url = get_phoenix_url()
            print(f"\nPhoenix observability enabled: {phoenix_url}\n")
        else:
            print("\nPhoenix initialization failed. Continuing without observability.\n")

    singlehop_model = SingleHopQA(
        retriever=base_retriever,
        chat_model="gpt-4.1-mini",
        temperature=0.0,
    )

    multihop_model = MultiHopQA(
        retriever=base_retriever,
        chat_model="gpt-4.1-mini",
        temperature=0.0,
        max_hops=3,
        max_docs_per_hop=3,
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

    print("\n==================================================================")
    print("SINGLE-HOP RAG")
    single_hop_pred = singlehop_model.predict(question, k=k_retrieve, trace=True)
    print("EM:", exact_match_score(single_hop_pred, answer))
    print("F1:", f1_score(single_hop_pred, answer))
    print("LLM Eval:", llm_eval_score(question, answer, single_hop_pred)["score"])

    print("\n==================================================================")
    print("MULTI-HOP RAG")
    multi_hop_pred = multihop_model.predict(question, k=k_retrieve, trace=True)
    print("EM:", exact_match_score(multi_hop_pred, answer))
    print("F1:", f1_score(multi_hop_pred, answer))
    llm_eval = llm_eval_score(question, answer, multi_hop_pred)
    print("LLM Eval:", llm_eval["score"])
    print("LLM Eval Explanation:", llm_eval["explanation"]) # Optional

    # Shutdown Phoenix if it was enabled
    if args.phoenix:
        print("\nFlushing traces to Phoenix...")
        from utils.phoenix_config import get_phoenix_manager
        manager = get_phoenix_manager()
        if manager and manager.tracer_provider:
            manager.tracer_provider.force_flush(timeout_millis=5000)
        shutdown_phoenix()

if __name__ == "__main__":
    main()