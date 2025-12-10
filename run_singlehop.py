import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from retrievers.faiss_retriever import FaissRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker, RerankRetriever
from models.single_hop import SingleHopQA
from utils.eval import evaluate_qa_system
from utils.phoenix_config import initialize_phoenix, get_phoenix_url, shutdown_phoenix
from datasets.load import json

def load_validation_set(name):
    if name == "hotpot":
        return load_dataset("hotpot_qa", "fullwiki", split="validation")
    if name == "musique":
        return load_dataset("dgslibisey/MuSiQue", split="validation")
    if name == "2wiki":
        with open("2wiki/dev.json", "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unknown dataset: {name}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()

    parser.add_argument("--retrieval_mode", type=str, default="faiss", help="faiss, bm25, hybrid")
    parser.add_argument("--dataset_name", type=str, default="hotpot", help="hotpot, musique, 2wiki")
    parser.add_argument("--k_retrieve", type=int, default=5)
    parser.add_argument("--n_eval", type=int, default=100)
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
    n_eval = args.n_eval
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
        project_name = f"{dataset_name}_{effective_retriever_name}_singlehop"
        tracer_provider = initialize_phoenix(
            project_name=project_name,
            port=args.phoenix_port,
            auto_instrument=True,
        )
        if tracer_provider:
            phoenix_url = get_phoenix_url()
            print(f"\n📊 Phoenix observability enabled: {phoenix_url}\n")
        else:
            print("\n⚠️  Phoenix initialization failed. Continuing without observability.\n")

    model = SingleHopQA(
        retriever=base_retriever,
        chat_model="gpt-4.1-mini",
        temperature=0.0,
    )

    ds_val = load_validation_set(dataset_name)

    def predict_fn(question, k):
        return model.predict(question, k=k)

    metrics = evaluate_qa_system(ds_val, predict_fn, n=n_eval, k=k_retrieve)

    print(f"Dataset: {dataset_name}")
    print(f"Retriever: {effective_retriever_name}")
    print("Results:", metrics)

    # Shutdown Phoenix if it was enabled
    if args.phoenix:
        shutdown_phoenix()

if __name__ == "__main__":
    main()