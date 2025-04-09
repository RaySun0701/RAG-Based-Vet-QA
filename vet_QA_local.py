import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader

from llama_index.core.settings import Settings
from llama_index.core.llms import LLM, CompletionResponse
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.storage import StorageContext
import torch

# === Configurable Paths ===
QWEN_MODEL_PATH = "/Users/raysun/Desktop/RAG-Based-Vet-QA/qwen_0.5b"
BGE_MODEL_PATH = "/Users/raysun/Desktop/RAG-Based-Vet-QA/bge_retriever"
CHUNKED_FILE = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_corpus/chunked_merck.jsonl"
INDEX_DIR = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_local_embedding"

# === HuggingFace LLM Wrapper (Replaces Local Qwen) ===
from llama_index.llms.huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0,
        "do_sample": False
    },
    tokenizer_name=QWEN_MODEL_PATH,
    model_name=QWEN_MODEL_PATH,
    device_map="auto"
)

# === HuggingFace Embedding Wrapper (Replaces Local BGE) ===
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name=BGE_MODEL_PATH,
    cache_folder=BGE_MODEL_PATH
)


# === Load chunked JSONL ===
def load_documents_from_jsonl(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docs.append(Document(
                text=item["content"],
                metadata={
                    "title": item.get("title", ""),
                    "section": item.get("section", ""),
                    "url": item.get("url", "")
                }
            ))
    return docs

# === Main Pipeline ===
def build_index():
    print("Loading documents...")
    documents = load_documents_from_jsonl(CHUNKED_FILE)

    print("Initializing LLM and Embedding models...")
    # Replaced with HuggingFaceLLM above
    # llm = LocalQwenLLM(QWEN_MODEL_PATH)
    # Replaced with HuggingFaceEmbedding above
    # embed_model = LocalBGEEmbedding(BGE_MODEL_PATH)
    # Simplified context setup using built-in API
    from llama_index.core import Settings
    # Use built-in setting without Pydantic conflict
    # from llama_index.llms.base import LLM as LLMType  # Deprecated or moved
    Settings.llm = llm  # type: ignore[arg-type]
    # from llama_index.embeddings.base import BaseEmbedding as EmbedType  # Deprecated or moved
    Settings.embed_model = embed_model  # type: ignore[arg-type]

    print("Building index...")
    index = VectorStoreIndex.from_documents(documents, )
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("Index saved to", INDEX_DIR)
    return index


def load_or_create_index():
    index_path = Path(INDEX_DIR)
    if index_path.exists() and any(index_path.glob("*.json")):
        from llama_index.core import load_index_from_storage
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)
    else:
        return build_index()


def interactive_query(index):
    from llama_index.core.prompts import PromptTemplate

    print("\nVet QA System Ready. Type your question, or 'exit' to quit.")

    system_prompt = (
        "You are a professional veterinary assistant. Use context from the veterinary literature to help answer user questions."
    )

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=system_prompt,
        memory=None,
        similarity_top_k=8
    )

    while True:
        query = input("\n> ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        response = chat_engine.chat(query)
        print("\nAnswer:\n", str(response))

        # Show sources
        if response.source_nodes:
            print("\nSources:")
            for i, node in enumerate(response.source_nodes, 1):
                meta = node.metadata
                title = meta.get("title", "Untitled")
                url = meta.get("url", "")
                print(f"[{i}] {title}\n    {url}")


if __name__ == "__main__":
    index = load_or_create_index()
    interactive_query(index)
