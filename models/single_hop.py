import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from retrievers.reranker import CrossEncoderReranker, RerankRetriever

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

        docs = self.retriever.similarity_search(question, k=k)

        if trace:
            print("\nSINGLE-HOP TRACE")
            for i, d in enumerate(docs, 1):
                snippet = d.page_content[:200].replace("\n", " ")
                print(f"[{i}] {snippet}...   meta={d.metadata}")

        context = "\n\n".join([d.page_content for d in docs])
        msg = [
            {"role": "system", "content": "Answer using only given context."},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}
        ]
        answer = self.llm.invoke(msg).content.strip()

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
