import os
import sys
import time
from datasets import load_dataset
from dotenv import load_dotenv
from utils.chunking import chunk_corpus
from utils.jsonl_utils import save_corpus_jsonl
from retrievers.faiss_retriever import build_faiss_index

load_dotenv()

def load_hotpot_fullwiki():
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
    return corpus


def load_musique():
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
    return corpus


def load_2wiki():
    import json
    base = "2wiki"
    files = ["train.json", "dev.json"]

    corpus = []
    seen = set()

    for fname in files:
        path = os.path.join(base, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            for title, sents in row["context"]:
                text = " ".join(sents)
                key = (title.lower(), text.lower())
                if key not in seen:
                    seen.add(key)
                    corpus.append({"title": title, "text": text})
    return corpus

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

def main():
    ds_name, loader_fn = select_dataset()

    start = time.time()
    print(f"\n=== Building Vector Store for: {ds_name} ===")
    corpus = loader_fn()

    texts, metas = chunk_corpus(corpus)

    corpus_path = os.path.join("corpora", f"{ds_name}.jsonl")
    save_corpus_jsonl(texts, metas, corpus_path)

    output_dir = os.path.join("vector_stores", f"faiss_{ds_name}")
    build_faiss_index(texts, metas, output_dir=output_dir)

if __name__ == "__main__":
    main()
