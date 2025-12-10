import json
from typing import List, Dict, Any
from pydantic import Field, conlist, create_model
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from retrievers.reranker import CrossEncoderReranker, RerankRetriever
from utils.prompts import (
    DECOMPOSER_SYSTEM_PROMPT, DECOMPOSER_USER_TEMPLATE,
    COMPOSE_QUERY_SYSTEM_PROMPT, ANSWER_SUBQ_SYSTEM_PROMPT, FINAL_ANSWER_SYSTEM_PROMPT,
    build_compose_query_prompt, build_answer_subq_prompt, build_final_answer_prompt
)
from utils.phoenix_tracing import (
    trace_rag_pipeline,
    trace_multihop_decomposition,
    trace_multihop_hop,
    trace_multihop_synthesis
)
from utils.phoenix_config import is_phoenix_enabled, MULTIHOP_INTERMEDIATE_ANSWER, LLM_OPERATION


class MultiHopQA:

    def __init__(
        self,
        retriever,
        chat_model="gpt-4.1-mini",
        temperature=0.0,
        max_hops=3,
        max_docs_per_hop=3,
        use_reranker=True,
        prefetch_k=20,
    ):
        if use_reranker:
            try:
                rerank_engine = CrossEncoderReranker()
                self.retriever = RerankRetriever(retriever, rerank_engine, prefetch_k=prefetch_k)
            except Exception:
                self.retriever = retriever
        else:
            self.retriever = retriever

        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)
        self.max_hops = max_hops
        self.max_docs_per_hop = max_docs_per_hop
        self.decomposer = self._build_decomposer(max_hops)

    def _build_decomposer(self, max_hops):
        Schema = create_model("Decompose", subquestions=(conlist(str, min_length=2, max_length=max_hops), Field(description="ordered reasoning hops")))
        structured = self.llm.bind_tools(tools=[], response_format=Schema, strict=True)

        def decompose(q):
            msgs = [
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user",  "content": DECOMPOSER_USER_TEMPLATE.format(max_subqs=max_hops, q=q)}
            ]
            resp = structured.invoke(msgs)
            return resp.additional_kwargs["parsed"].subquestions

        return decompose

    def predict(self, question: str, k=5, trace=False, save_trace=None):
        with trace_rag_pipeline(question, "multi_hop") as pipeline_span:
            hops = []
            
            # Decomposition span
            with trace_multihop_decomposition(question, self.max_hops):
                subqs = self.decomposer(question)
            
            if pipeline_span and is_phoenix_enabled():
                pipeline_span.set_attribute("multihop.num_subquestions", len(subqs))

            # Per-hop processing
            for hop_idx, subq in enumerate(subqs, 1):
                # Query composition (if not first hop)
                if hops:
                    composed = self.llm.invoke([
                        {"role":"system","content":COMPOSE_QUERY_SYSTEM_PROMPT},
                        {"role":"user", "content":build_compose_query_prompt(question, subq, hops)}
                    ]).content.strip()
                else:
                    composed = subq

                # Hop span with nested retrieval and answer generation
                with trace_multihop_hop(hop_idx, subq, composed) as hop_span:
                    docs = self.retriever.similarity_search(composed, k=k)[:self.max_docs_per_hop]

                    passages = [d.page_content for d in docs]
                    answer = self.llm.invoke([
                        {"role":"system","content":ANSWER_SUBQ_SYSTEM_PROMPT},
                        {"role":"user","content":build_answer_subq_prompt(question, subq, passages, hops)}
                    ]).content.strip()

                    # Add hop attributes
                    if hop_span and is_phoenix_enabled():
                        hop_span.set_attribute(MULTIHOP_INTERMEDIATE_ANSWER, answer)
                        hop_span.set_attribute("multihop.num_docs", len(docs))

                if trace:
                    print(f"\nHOP {hop_idx}")
                    print("Subquestion:", subq)
                    print("Composed question:", composed)
                    print("\nPassages:")
                    for i,d in enumerate(docs,1):
                        print(f"[{i}] {d.page_content[:200]}... meta={d.metadata}")
                    print(f"Predicted Answer: {answer}\n")

                hops.append({
                    "subq": subq,
                    "composed": composed,
                    "docs": passages,
                    "answer": answer
                })

            # Final synthesis span
            with trace_multihop_synthesis(question, len(hops)):
                final = self.llm.invoke([
                    {"role":"system","content":FINAL_ANSWER_SYSTEM_PROMPT},
                    {"role":"user","content":build_final_answer_prompt(question, hops)}
                ]).content.strip()

            # Add final answer to pipeline span
            if pipeline_span and is_phoenix_enabled():
                pipeline_span.set_attribute("rag.answer", final)
                pipeline_span.set_attribute("rag.total_hops", len(hops))

            if trace:
                print("\nOriginal Question:", question)
                print("Predicted Final Answer:", final) 

            if save_trace:
                with open(save_trace, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question":question,"hops":hops,"final":final})+"\n")
                print(f"Trace saved to {save_trace}")

            return final