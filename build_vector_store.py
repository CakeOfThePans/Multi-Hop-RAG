import os
import sys
import time
from datasets import load_dataset
from dotenv import load_dotenv

# Local imports
from utils.chunking import chunk_corpus
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
    print("Loading MuSiQue (full_v1)…")
    ds = load_dataset("musique", "full_v1")["train"]

    corpus = []
    seen = set()

    for row in ds:
        for p in row["paragraphs"]:
            title = p["title"]
            text = " ".join(p["sentences"])
            key = (title.lower(), text.lower())
            if key not in seen:
                seen.add(key)
                corpus.append({"title": title, "text": text})

    print(f"Collected {len(corpus)} MuSiQue paragraphs.")
    return corpus


def load_2wiki():
    print("Loading 2WikiMultiHopQA (main)…")
    ds = load_dataset("2wikimultihopqa", "main")["train"]

    corpus = []
    seen = set()

    for row in ds:
        for title, sentences in row["context"]:
            text = " ".join(sentences)
            key = (title.lower(), text.lower())
            if key not in seen:
                seen.add(key)
                corpus.append({"title": title, "text": text})

    print(f"Collected {len(corpus)} 2Wiki paragraphs.")
    return corpus


# -------------------------------
# Menu Selection
# -------------------------------

def select_dataset():
    print("\n=== VECTOR STORE BUILDER ===")
    print("Select dataset to embed:\n")
    print("[1] HotpotQA – fullwiki")
    print("[2] MuSiQue – full_v1")
    print("[3] 2WikiMultiHopQA – main")
    print("[0] Cancel\n")

    choice = input("Enter selection: ").strip()

    if choice == "1":
        return "hotpot_fullwiki", load_hotpot_fullwiki
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
    chunks = chunk_corpus(corpus)

    # 3. Build FAISS index
    print("\nBuilding FAISS vector store…")
    output_dir = os.path.join("vector_stores", f"faiss_{ds_name}")
    build_faiss_index(chunks, output_dir=output_dir)

    elapsed = time.time() - start
    print(f"\n✔ DONE — Vector store built for: {ds_name}")
    print(f"⏱ Total time: {elapsed/60:.2f} minutes")
    print(f"📌 Saved under: {output_dir}/\n")


if __name__ == "__main__":
    main()
