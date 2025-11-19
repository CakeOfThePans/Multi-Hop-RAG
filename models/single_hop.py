from typing import List
from langchain_openai import ChatOpenAI

from utils.prompts import SINGLE_HOP_SYSTEM_PROMPT, build_singlehop_user_prompt


class SingleHopQA:
    """
    Single-hop RAG system: retrieve top-k passages, feed into LLM, output answer.
    """

    def __init__(self, retriever, chat_model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)

    def predict(self, question: str, k: int = 5) -> str:
        docs = self.retriever.similarity_search(question, k=k)
        passages: List[str] = [d.page_content for d in docs]
        user_prompt = build_singlehop_user_prompt(question, passages)
        resp = self.llm.invoke(
            [
                {"role": "system", "content": SINGLE_HOP_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return resp.content.strip()