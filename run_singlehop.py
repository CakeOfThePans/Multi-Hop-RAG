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
            print(f"\nPhoenix observability enabled: {phoenix_url}\n")
        else:
            print("\nPhoenix initialization failed. Continuing without observability.\n")

    model = SingleHopQA(
        retriever=base_retriever,
        chat_model="gpt-4.1-mini",
        temperature=0.0,
    )

    ds_val = load_validation_set(dataset_name)

    def predict_fn(question, k):
        return model.predict(question, k=k)

    # Evaluate with Phoenix tracing and error categorization
    metrics = evaluate_qa_system(
        ds_val,
        predict_fn,
        n=n_eval,
        k=k_retrieve,
        dataset_name=dataset_name,
        retrieval_mode=effective_retriever_name,
        architecture="single_hop",
        export_data=args.phoenix,  # Export data if Phoenix is enabled
    )

    print(f"Dataset: {dataset_name}")
    print(f"Retriever: {effective_retriever_name}")
    print("Results:", metrics)
    
    # Export evaluation data to Phoenix if enabled
    if args.phoenix and "eval_data" in metrics:
        from utils.phoenix_export import export_evaluation_to_phoenix, export_evaluation_summary
        
        eval_data = metrics["eval_data"]
        df = export_evaluation_to_phoenix(
            questions=eval_data["questions"],
            predictions=eval_data["predictions"],
            ground_truths=eval_data["ground_truths"],
            em_scores=eval_data["em_scores"],
            f1_scores=eval_data["f1_scores"],
            llm_scores=eval_data["llm_scores"],
            error_categories=eval_data["error_categories"],
            dataset_name=dataset_name,
            retrieval_mode=effective_retriever_name,
            architecture="single_hop",
            llm_explanations=eval_data.get("llm_explanations"),
        )
        
        if df is not None:
            summary = export_evaluation_summary(
                eval_data,
                dataset_name,
                effective_retriever_name,
                "single_hop",
            )
            print(f"\nError Analysis Summary:")
            print(f"   Total Samples: {summary['total_samples']}")
            print(f"   Correct: {summary['correct']} ({100*(1-summary['error_rate']):.1f}%)")
            print(f"   Error Rate: {100*summary['error_rate']:.1f}%")
            print(f"   Top Error Category: {summary['top_error_category']}")
            print(f"   Error Distribution: {summary['error_distribution']}")

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