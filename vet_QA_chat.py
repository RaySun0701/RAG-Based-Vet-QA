import os
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Paths ===
QWEN_MODEL_PATH = "/Users/raysun/Desktop/RAG-Based-Vet-QA/qwen_0.5b"
BGE_MODEL_PATH = "/Users/raysun/Desktop/RAG-Based-Vet-QA/bge_retriever"
INDEX_DIR = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_local_embedding"

# === Initialize local HuggingFace LLM ===
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

# === Initialize local embedding model ===
embed_model = HuggingFaceEmbedding(
    model_name=BGE_MODEL_PATH,
    cache_folder=BGE_MODEL_PATH
)

# === Register LLM and embedding model ===
Settings.llm = llm
Settings.embed_model = embed_model

# === Load index ===
print("Loading vector index...")
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)

# === Start chat engine ===
from llama_index.core.prompts import PromptTemplate

system_prompt = (
    "You are a professional veterinary assistant. Use context from the veterinary literature to help answer user questions."
)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=system_prompt,
    memory=None,
    similarity_top_k=8
)

# === Interactive chat loop ===
print("\nVet QA System Ready. Type your question (or 'exit' to quit).")

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
