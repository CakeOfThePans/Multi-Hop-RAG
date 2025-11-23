from typing import List, Dict, Any
from pydantic import Field, conlist, create_model
from langchain_openai import ChatOpenAI

from utils.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_USER_TEMPLATE,
    COMPOSE_QUERY_SYSTEM_PROMPT,
    ANSWER_SUBQ_SYSTEM_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    build_compose_query_prompt,
    build_answer_subq_prompt,
    build_final_answer_prompt,
)


class MultiHopQA:
    """
    Multi-hop RAG system:
      - LLM-based question decomposition
      - iterative retrieval & answering
      - final answer synthesis
    """

    def __init__(self, retriever, chat_model: str = "gpt-4o-mini", temperature: float = 0.0, max_hops: int = 3, max_docs_per_hop: int = 3):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)
        self.max_hops = max_hops
        self.max_docs_per_hop = max_docs_per_hop
        self.decomposer = self._make_decomposer(max_subqs=max_hops)

    # ---------- Decomposition ----------

    def _make_decomposer(self, max_subqs: int):
        DecompSchema = create_model(
            "DecompSchema",
            subquestions=(
                conlist(str, min_length=2, max_length=max_subqs),
                Field(
                    description=(
                        "Ordered sub-questions to solve the original in sequence."
                    )
                ),
            ),
        )

        structured_llm = self.llm.bind_tools(tools=[], response_format=DecompSchema, strict=True)

        def decompose(question: str) -> List[str]:
            user_prompt = DECOMPOSER_USER_TEMPLATE.format(
                max_subqs=max_subqs, q=question
            )
            msgs = [
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            resp = structured_llm.invoke(msgs)
            parsed = resp.additional_kwargs["parsed"]
            return parsed.subquestions

        return decompose

    # ---------- Helper LLM calls ----------

    def _compose_query(self, question: str, subq: str, hops: List[Dict[str, Any]]) -> str:
        user_prompt = build_compose_query_prompt(question, subq, hops)
        resp = self.llm.invoke(
            [
                {"role": "system", "content": COMPOSE_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return resp.content.strip()

    def _answer_subq(self, question: str, subq: str, passages: List[str], hops: List[Dict[str, Any]]) -> str:
        user_prompt = build_answer_subq_prompt(question, subq, passages, hops)
        resp = self.llm.invoke(
            [
                {"role": "system", "content": ANSWER_SUBQ_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return resp.content.strip()

    def _get_final_answer(self, question: str, hops: List[Dict[str, Any]]) -> str:
        user_prompt = build_final_answer_prompt(question, hops)
        resp = self.llm.invoke(
            [
                {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return resp.content.strip()

    # ---------- Main pipeline ----------

    def predict(self, question: str, k: int = 5) -> str:
        subquestions = self.decomposer(question)
        hops: List[Dict[str, Any]] = []

        for subq in subquestions:
            if hops:
                composed_q = self._compose_query(question, subq, hops)
            else:
                composed_q = subq

            retrieved_docs = self.retriever.similarity_search(composed_q, k=k)
            passages = [d.page_content for d in retrieved_docs][:self.max_docs_per_hop]

            ans = self._answer_subq(question, subq, passages, hops)

            hops.append(
                {
                    "subq": subq,
                    "composed": composed_q,
                    "passages": passages,
                    "answer": ans,
                }
            )

        final_answer = self._get_final_answer(question, hops)
        return final_answer