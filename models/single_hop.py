from typing import List
from langchain_openai import ChatOpenAI

from utils.prompts import SINGLE_HOP_SYSTEM_PROMPT, build_singlehop_user_prompt


class SingleHopQA:
    """
    Single-hop RAG system: retrieve top-k passages, feed into LLM, output answer.
    """

    def __init__(self, retriever, chat_model: str = "gpt-4o-mini", temperature: float = 0.0, verbose: bool = False):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=chat_model, temperature=temperature)
        self.verbose = verbose

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

        if self.verbose:
            for i, p in enumerate(passages):
                print(f"Passage {i + 1}:\n{p}\n")
            print(f"Predicted Answer:", resp.content.strip())

        return resp.content.strip()