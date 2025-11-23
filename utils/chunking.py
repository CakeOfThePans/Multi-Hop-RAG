from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_corpus(
    corpus_rows: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Tuple[List, List]:
    """
    Chunk the corpus rows into (text, metadata) tuples.
    This is the format required by the FAISS builder:
        [ (chunk_text, {"title": ...}), ... ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    texts, metas = [], []
    for r in corpus_rows:
        chunks = splitter.split_text(r['text'])
        texts.extend(chunks)
        metas.extend([{"title": r['title']} for _ in chunks])

    return texts, metas

