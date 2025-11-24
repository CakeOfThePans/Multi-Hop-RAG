import json
import os
from langchain_core.documents import Document

def save_corpus_jsonl(texts, metas, path):
    os.makedirs("corpora", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, meta in zip(texts, metas):
            f.write(json.dumps({
                "text": text,
                "title": meta.get("title", "UNKNOWN")
            }) + "\n")

def load_corpus_from_jsonl(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(page_content=obj["text"], metadata={"title": obj["title"]}))
    return docs