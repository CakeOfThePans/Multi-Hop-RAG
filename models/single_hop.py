import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from retrievers.reranker import CrossEncoderReranker, RerankRetriever
from utils.prompts import SINGLE_HOP_SYSTEM_PROMPT, build_singlehop_user_prompt
from utils.phoenix_tracing import trace_rag_pipeline
from utils.phoenix_config import is_phoenix_enabled

class SingleHopQA:

    def __init__(
        self,
        retriever,
        chat_model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        use_reranker: bool = True,
        prefetch_k: int = 20,
    ):

        if use_reranker:
            try:
                rerank_engine = CrossEncoderReranker()
                self.retriever = RerankRetriever(retriever, rerank_engine, prefetch_k=prefetch_k)
            except Exception as e:
                print(e)
                self.retriever = retriever
        else:
            self.retriever = retriever
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)

    def predict(self, question, k = 5, trace= False, save_trace = None):
        with trace_rag_pipeline(question, "single_hop") as pipeline_span:
            docs = self.retriever.similarity_search(question, k=k)

            passages = [d.page_content for d in docs]
            user_prompt = build_singlehop_user_prompt(question, passages)
            resp = self.llm.invoke(
                [
                    {"role": "system", "content": SINGLE_HOP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            answer = resp.content.strip()

            # Add attributes to pipeline span
            if pipeline_span and is_phoenix_enabled():
                pipeline_span.set_attribute("rag.num_documents", len(docs))
                pipeline_span.set_attribute("rag.answer", answer)
                if docs:
                    pipeline_span.set_attribute("rag.top_doc_snippet", docs[0].page_content[:200].replace("\n", " "))

            if trace:
                print("\nPassages:")
                for i, d in enumerate(docs, 1):
                    snippet = d.page_content[:200].replace("\n", " ")
                    print(f"[{i}] {snippet}...   meta={d.metadata}")
                print(f"Predicted Answer: {answer}\n")

            if save_trace:
                with open(save_trace, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "question": question,
                        "answer": answer,
                        "docs": [d.page_content for d in docs],
                        "meta": [d.metadata for d in docs],
                    }) + "\n")
                print(f"Trace written to {save_trace}")

            return answer
