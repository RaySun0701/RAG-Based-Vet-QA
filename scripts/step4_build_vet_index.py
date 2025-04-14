import json
from pathlib import Path
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Base directory: root of your project ===
BASE_DIR = Path(__file__).resolve().parents[1]  # Goes up from /scripts/

# === Paths relative to project structure ===
CHUNKED_FILE = BASE_DIR / "vet_corpus" / "chunked_merck.jsonl"
INDEX_DIR = BASE_DIR / "vet_local_embedding"
BGE_MODEL_PATH = BASE_DIR / "bge_retriever"

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
        model_name=str(BGE_MODEL_PATH),
        cache_folder=str(BGE_MODEL_PATH)
    )
    Settings.embed_model = embed_model

    print("Step 3: Building index...")
    index = VectorStoreIndex.from_documents(documents)

    print("Step 4: Saving index to disk...")
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    print(f"Index saved to: {INDEX_DIR}")

if __name__ == "__main__":
    build_index()

