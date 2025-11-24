import os
import sys
import time
from datasets import load_dataset
from dotenv import load_dotenv

# Local imports
from utils.chunking import chunk_corpus
from utils.jsonl_utils import save_corpus_jsonl
from retrievers.faiss_retriever import build_faiss_index

load_dotenv()

# -------------------------------
# Dataset Loaders
# -------------------------------

def load_hotpot_fullwiki():
    print("Loading HotpotQA (fullwiki split)…")
    ds_train = load_dataset("hotpot_qa", "fullwiki", split="train")
    ds_val = load_dataset("hotpot_qa", "fullwiki", split="validation")

    corpus = []
    seen = set()

    for split in [ds_train, ds_val]:
        for row in split:
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
                text = " ".join(sents)
                key = (title.lower(), text.lower())
                if key not in seen:
                    seen.add(key)
                    corpus.append({"title": title, "text": text})

    print(f"Collected {len(corpus)} paragraphs from HotpotQA.")
    return corpus


def load_musique():
    print("Loading MuSiQue…")
    ds_train = load_dataset("dgslibisey/MuSiQue", split="train")
    ds_val = load_dataset("dgslibisey/MuSiQue", split="validation")

    corpus = []
    seen = set()

    for split in [ds_train, ds_val]:
        for row in split:
            for paragraph in row["paragraphs"]:
                key = (paragraph["title"].lower(), paragraph["paragraph_text"].lower())
                if key not in seen:
                    seen.add(key)
                    corpus.append({"title": paragraph["title"], "text": paragraph["paragraph_text"]})

    print(f"Collected {len(corpus)} paragraphs from MuSiQue.")
    return corpus


def load_2wiki():
    print("Loading 2WikiMultiHopQA…")
    ds_train = load_dataset("framolfese/2WikiMultihopQA", split="train")
    ds_val = load_dataset("framolfese/2WikiMultihopQA", split="validation")

    corpus = []
    seen = set()

    for split in [ds_train, ds_val]:
        for row in split:
            for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
                text = " ".join(sents)
                key = (title.lower(), text.lower())
                if key not in seen:
                    seen.add(key)
                    corpus.append({"title": title, "text": text})

    print(f"Collected {len(corpus)} paragraphs from 2WikiMultiHopQA.")
    return corpus


# -------------------------------
# Menu Selection
# -------------------------------

def select_dataset():
    print("\n=== VECTOR STORE BUILDER ===")
    print("Select dataset to embed:\n")
    print("[1] HotpotQA")
    print("[2] MuSiQue")
    print("[3] 2WikiMultiHopQA")
    print("[0] Cancel\n")

    choice = input("Enter selection: ").strip()

    if choice == "1":
        return "hotpot", load_hotpot_fullwiki
    elif choice == "2":
        return "musique", load_musique
    elif choice == "3":
        return "2wiki", load_2wiki
    elif choice == "0":
        print("Cancelled.")
        sys.exit(0)
    else:
        print("Invalid choice.")
        sys.exit(1)


# -------------------------------
# Main Entrypoint
# -------------------------------

def main():
    ds_name, loader_fn = select_dataset()

    start = time.time()
    print(f"\n=== Building Vector Store for: {ds_name} ===")

    # 1. Load corpus
    corpus = loader_fn()

    # 2. Chunk corpus
    print("\nChunking corpus…")
    texts, metas = chunk_corpus(corpus)

    # 3. Save corpus as jsonl
    corpus_path = os.path.join("corpora", f"{ds_name}.jsonl")
    save_corpus_jsonl(texts, metas, corpus_path)
    print(f"Saved corpus in corpora/{ds_name}.jsonl")

    # # 4. Build FAISS index
    print("\nBuilding FAISS vector store…")
    output_dir = os.path.join("vector_stores", f"faiss_{ds_name}")
    build_faiss_index(texts, metas, output_dir=output_dir)

    elapsed = time.time() - start
    print(f"\n✔ DONE — Vector store built for: {ds_name}")
    print(f"⏱ Total time: {elapsed/60:.2f} minutes")
    print(f"📌 Saved under: {output_dir}/\n")


if __name__ == "__main__":
    main()
