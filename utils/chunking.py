import re
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import Dataset


def build_corpus_from_hotpotqa(ds_train: Dataset, ds_val: Dataset) -> List[Dict]:
    """
    Build a paragraph-level corpus from HotpotQA train + val contexts,
    then deduplicate paragraphs.
    """
    corpus_rows = []

    # Train
    for example in ds_train:
        titles = example["context"]["title"]
        sentences_lists = example["context"]["sentences"]
        for title, sents in zip(titles, sentences_lists):
            paragraph_text = " ".join(sents)
            corpus_rows.append({"title": title, "text": paragraph_text})

    # Validation
    for example in ds_val:
        titles = example["context"]["title"]
        sentences_lists = example["context"]["sentences"]
        for title, sents in zip(titles, sentences_lists):
            paragraph_text = " ".join(sents)
            corpus_rows.append({"title": title, "text": paragraph_text})

    # Deduplicate by (title, cleaned_text)
    unique_seen = set()
    unique_rows = []
    for row in corpus_rows:
        clean_text = re.sub(r"\s+", " ", row["text"]).strip().lower()
        key = (row["title"], clean_text)
        if key not in unique_seen:
            unique_seen.add(key)
            unique_rows.append({"title": row["title"], "text": row["text"]})

    return unique_rows


def chunk_corpus(
    corpus_rows: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Tuple[str, Dict]]:
    """
    Chunk the corpus rows into (text, metadata) tuples.
    This is the format required by the FAISS builder:
        [ (chunk_text, {"title": ...}), ... ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_tuples = []

    for r in corpus_rows:
        chunks = splitter.split_text(r["text"])
        for c in chunks:
            chunk_tuples.append((c, {"title": r["title"]}))

    return chunk_tuples

