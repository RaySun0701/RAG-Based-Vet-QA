import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer

# === Paths ===
QWEN_MODEL_PATH = "/Users/raysun/Desktop/RAG-Based-Vet-QA/qwen_0.5b"       
BGE_MODEL_PATH = "/Users/raysun/Desktop/RAG-Based-Vet-QA/bge_retriever"
INDEX_DIR = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_local_embedding"

# === Initialize local HuggingFace LLM ===
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={
        "do_sample": False,  # deterministic output
        "top_p": None        # disable nucleus sampling
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

# === Define veterinary-style system prompt ===
system_prompt = (
    "You are a professional veterinary assistant. Use context from the veterinary literature to help answer user questions."
)

# === Start chat engine ===
memory = ChatMemoryBuffer.from_defaults()
chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=system_prompt,
    memory=memory,  # Use memory buffer
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

    # Show sources (deduplicated by URL)
    if response.source_nodes:
        print("\nSources:")
        seen_urls = set()
        i = 1
        for node in response.source_nodes:
            meta = node.metadata
            url = meta.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = meta.get("title", "Untitled")
            print(f"[{i}] {title}\n    {url}")
            i += 1