import re
from typing import List, Dict, Tuple
from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_corpus_from_hotpotqa(ds_train, ds_val):
    rows = []
    for ex in ds_train:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            rows.append({"title": title, "text": " ".join(sents)})

    for ex in ds_val:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            rows.append({"title": title, "text": " ".join(sents)})

    unique = set()
    cleaned = []
    for r in rows:
        clean = re.sub(r"\s+", " ", r["text"]).strip().lower()
        key = (r["title"].lower(), clean)
        if key not in unique:
            unique.add(key)
            cleaned.append({"title": r["title"], "text": r["text"]})

    return cleaned



def chunk_corpus(corpus_rows, chunk_size = 500, chunk_overlap = 50):
    """
    Converts full paragraphs into (text_chunk, metadata) tuples.
    Output format supports FAISS + BM25 + Hybrid + reranker.

    Returns:
        texts -> [chunk1_text, chunk2_text, ...]
        metas -> [{"title": ...}, ...] aligned with texts
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    texts, metas = [], []
    for r in corpus_rows:
        chunks = splitter.split_text(r["text"])
        texts.extend(chunks)
        metas.extend([{"title": r["title"]} for _ in chunks])

    return texts, metas