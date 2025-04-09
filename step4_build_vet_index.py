import json
from pathlib import Path
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Configurable Paths ===
CHUNKED_FILE = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_corpus/chunked_merck.jsonl"   # Path to the chunked JSONL file
INDEX_DIR = "/Users/raysun/Desktop/vet_local_embedding"                 # Path to save the index
BGE_MODEL_PATH = "/Users/raysun/Desktop/bge_retriever"                  # Path to the BGE model

# === Step 1: Load JSONL documents ===
def load_documents_from_jsonl(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            item = json.loads(line)
            documents.append(Document(
                text=item["content"],
                metadata={
                    "title": item.get("title", ""),
                    "section": item.get("section", ""),
                    "url": item.get("url", "")
                }
            ))
    return documents

# === Step 2: Build index using embeddings only ===
def build_index():
    print("Step 1: Loading documents...")
    documents = load_documents_from_jsonl(CHUNKED_FILE)

    print("Step 2: Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name=BGE_MODEL_PATH,
        cache_folder=BGE_MODEL_PATH
    )
    Settings.embed_model = embed_model

    print("Step 3: Building index...")
    index = VectorStoreIndex.from_documents(documents)

    print("Step 4: Saving index to disk...")
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"Index saved to: {INDEX_DIR}")

if __name__ == "__main__":
    build_index()
