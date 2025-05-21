# LBRXCHAT Veterinary RAG System

This repository combines two powerful components:
1. Ray Sun's Veterinary QA RAG System
2. LIBRAXIS Chat Framework (LBRXCHAT)

## Project Overview

The integrated system provides a **local Retrieval-Augmented Generation (RAG)** pipeline optimized for veterinary content, with an advanced terminal user interface. The system runs entirely locally, with no need for internet connection once models are downloaded.

### Key Features

- **Local MLX Models**: Optimized for Apple Silicon (M1/M2/M3)
- **Rich TUI Interface**: Beautiful terminal-based UI built with Textual
- **Veterinary Knowledge Base**: Pre-built with Merck Veterinary Manual content
- **JIT Model Loading**: Models loaded on demand with TTL for resource efficiency
- **LM Studio Integration**: Supports both native API and REST interfaces

### Supported Models

- **Embedding Models**: 
  - BAAI/bge-small-en-v1.5 (original)
  - Nomic Embed (via LM Studio)

- **Language Models**:
  - Qwen/Qwen2.5-0.5B-Instruct (original)
  - Qwen3 (8B/14B/32B via MLX)
  - Llama3 (8B/70B via MLX)
  - Mixtral (8x7B via MLX)
  - Mistral (7B via MLX)
  - Phi-3 (3.8B/14B via MLX)

## Installation

### Requirements

- Python 3.10+ (3.12 recommended)
- macOS with Apple Silicon (M1/M2/M3)
- LM Studio with MLX models (optional)
- 16GB+ RAM

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/libraxis/lbrxchat-vet-rag.git
cd lbrxchat-vet-rag

# Create virtual environment
uv venv -p 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Quick Start with LM Studio (Recommended)

1. Start LM Studio and load an MLX model (e.g., Qwen3-8B-MLX)
2. Run the TUI:

```bash
python -m lbrxchat.ui.tui
```

### Running with Local Models

Follow the original setup to download models and build indexes:

```bash
# Download BAAI/bge-small-en-v1.5 embedding model
python scripts/step3_download_bge_retriever.py

# Download Qwen/Qwen2.5-0.5B-Instruct language model
python scripts/step5_download_qwen_model.py

# Build the vector index
python scripts/step4_build_vet_index.py
```

### Sample Questions

```
> My dog vomited this morning. What could be the cause?
> What are common skin conditions in cats?
> How should I feed my pet turtle?
> What vaccines do puppies need?
> Can reptiles show signs of pain?
```

## Architecture

```
lbrxchat/
├── core/               # Core components
│   ├── rag.py          # RAG system
│   ├── embedding.py    # Embedding handling
│   ├── models.py       # Model management
│   └── config.py       # Configuration
├── ui/                 # User interface
│   ├── tui.py          # Main TUI interface
│   ├── components/     # UI components
│   └── styles.css      # CSS styles
├── data/               # Data management
│   ├── corpus.py       # Corpus handling
│   ├── index.py        # Index management
│   └── vector_store.py # Vector database
└── tools/              # Helper tools
    ├── build_index.py  # Index building
    ├── convert.py      # Data conversion
    └── benchmark.py    # Performance tests

scripts/                # Original processing scripts
vet_corpus/             # Paragraph-level content
vet_knowledge/          # Raw crawled data
vet_local_embedding/    # Prebuilt FAISS index
```

📍Saved to: `RAG-Based-Vet-QA/vet_knowledge/merck_knowledge.jsonl`

------

### Original Directory Structure

```
RAG-Based-Vet-QA/
├── bge_retriever/                    # Local BGE embedding model (download)
├── qwen_0.5b/                        # Local Qwen LLM (download)
├── scripts/                          # All processing scripts
│   ├── step1_vet_merck_scraper.py           # Crawl Merck Veterinary Manual
│   ├── step2_chunk_merck_by_paragraph.py    # Split content into paragraphs
│   ├── step3_download_bge_retriever.py      # Download BGE model
│   ├── step4_build_vet_index.py             # Build FAISS index 
│   ├── step5_download_qwen_model.py         # Download Qwen model
│   └── step6_vet_QA_chat.py                 # Run QA interface
├── vet_corpus/
│   └── chunked_merck.jsonl           # Paragraph-level content
├── vet_knowledge/
│   ├── merck_knowledge.jsonl         # Raw crawled data
│   └── visited_urls.json             # Visited URL cache
├── vet_local_embedding/              # Prebuilt FAISS index
├── requirements.txt
└── README.md
```

### Step-by-Step Pipeline

#### Step 1: Crawl Merck Veterinary Manual

**Script:** `scripts/step1_vet_merck_scraper.py`

The crawler extracts structured content (titles, sections, URLs, and clean paragraphs) from all educational articles hosted on the MVM website, serving as the foundation for the retrieval-based veterinary QA system.

#### Step 2: Chunk Content into Paragraphs

**Script:** `scripts/step2_chunk_merck_by_paragraph.py`

Processes the raw crawled content, splitting it into paragraph-sized chunks suitable for embedding and retrieval.

#### Step 3: Download Embedding Model

**Script:** `scripts/step3_download_bge_retriever.py`

Downloads the `BAAI/bge-small-en-v1.5` embedding model for local use.

#### Step 4: Build Vector Index

**Script:** `scripts/step4_build_vet_index.py`

Constructs a FAISS vector index from the paragraph chunks, enabling efficient semantic search.

#### Step 5: Download Language Model

**Script:** `scripts/step5_download_qwen_model.py`

Downloads the `Qwen/Qwen2.5-0.5B-Instruct` language model for local generation.

#### Step 6: Run QA Interface

**Script:** `scripts/step6_vet_QA_chat.py`

The original command-line interface for the QA system.

## License

MIT License

## Acknowledgments

- **Ray Sun** for the original Veterinary RAG system
- **LIBRAXIS Team** for the LBRXCHAT framework integration

