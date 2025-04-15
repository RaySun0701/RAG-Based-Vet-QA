

# Veterinary QA RAG System – Project Report

------

# Project Goal

Build a **local Retrieval-Augmented Generation (RAG)** pipeline that answers veterinary questions by retrieving and generating responses based on the **Merck Veterinary Manual** content using **local models**:

- Local embedding model: `BAAI/bge-small-en-v1.5`

- Local LLM: `Qwen/Qwen2.5-0.5B-Instruct`

### Notice: 
You can run the QA system on your local machine after downloading the models to your local machine, **no need to connect to internet**. This is a good setting for usage within the company, to keep your data safe.  
**The model used in this pipeline is replaceable.** I use a Macbook with M2 chip. It works well on my machine. If you have better hardware setting, you can definately use models with larger parameter size.  
All input/output paths are handled relative to the project root. No need to change any paths manually.

## Environment Setup

To set up and run this project locally, follow the steps below.

### 1. Clone the Repository

Choose the desired path you want for this repository.
```bash
cd /your/desired/path
git clone https://github.com/RaySun0701/RAG-Based-Vet-QA.git
cd RAG-Based-Vet-QA
```

### 2. (Optional But Recommended) Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
unalias python              # Optional: remove the alias if python points to system Python
which python                # Should point to the Python interpreter inside venv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

All dependencies used in the pipeline will be installed, including:

- `transformers`
- `sentence-transformers`
- `llama-index`
- `faiss-cpu`
- `tqdm`, `requests`, `beautifulsoup4`, etc.

### 4. Download Required Models

The following scripts will download the models to your local machine:

```bash
# Download BAAI/bge-small-en-v1.5 embedding model
python scripts/step3_download_bge_retriever.py

# Download Qwen/Qwen2.5-0.5B-Instruct language model
python scripts/step5_download_qwen_model.py
```

This will create:

- `bge_retriever/` – for the embedding model
- `qwen_0.5b/` – for the LLM

### 5. Build the Vector Index

If you are using `BAAI/bge-small-en-v1.5` embedding model, you need to build the vector index yourself.

Please run the code below:

```bash
python scripts/step4_build_vet_index.py
```
This will take a few minutes.

### 6. Run the QA System

```bash
python scripts/step6_vet_QA_chat.py
```

You can now interact with the system and ask veterinary-related questions.

### Sample Questions to Try

```text
> My dog vomited this morning. What could be the cause?
> What are common skin conditions in cats?
> How should I feed my pet turtle?
> What vaccines do puppies need?
> Can reptiles show signs of pain?
```

---

## Directory Structure

```
RAG-Based-Vet-QA/
├── bge_retriever/                    # Local BGE embedding model (download)
├── qwen_0.5b/                        # Local Qwen LLM (download)
├── scripts/                          # All processing scripts
│   ├── step1_vet_merck_scraper.py           # Crawl Merck Veterinary Manual
│   ├── step2_chunk_merck_by_paragraph.py    # Split content into paragraphs
│   ├── step3_download_bge_retriever.py      # Download BGE model
│   ├── step4_build_vet_index.py             # Build FAISS index (optional)
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

------

# Step-by-Step Pipeline

------

## 📌 Step 1: Crawl Merck Veterinary Manual

**Script:** `RAG-Based-Vet-QA/scripts/step1_vet_merck_scraper.py`

### Purpose:

The goal was to extract structured content (titles, sections, URLs, and clean paragraphs) from all educational articles hosted on the MVM website, which would later serve as the foundation for building a retrieval-based veterinary QA system.  
**Notice:** The crawler was carefully designed to comply with the website’s terms of use and crawling guidelines. Also **remember to change the output path to your own path in the code**.

### Example Output:

```json
{
  "title": "Description and Physical Characteristics of Amphibians",
  "section": "Veterinary > All Other Pets > Amphibians > Description and Physical Characteristics of Amphibians",
  "url": "https://www.merckvetmanual.com/all-other-pets/amphibians/description-and-physical-characteristics-of-amphibians",
  "paragraphs": ["...", "...", "..."]
}
```

📍Saved to: `RAG-Based-Vet-QA/vet_knowledge/merck_knowledge.jsonl`

------

## 📌 Step 2: Paragraph-based Chunking

**Script:** `RAG-Based-Vet-QA/scripts/step2_chunk_merck_by_paragraph.py`

### Purpose:

The main goal of this step is to segment each article into multiple **overlapping text chunks** while preserving paragraph boundaries and contextual continuity. This approach ensures that each chunk contains enough relevant information to support meaningful question-answering, without breaking sentence or paragraph structures.

### Technique:

- Sliding window:
  - Window size: 3
  - Step: 2
- Each chunk includes:
  - `chunk_id`
  - `article_id`
  - metadata (title, section, url)
  - `content`: merged paragraph text

**Assume a document has the following paragraphs:**

```text
para1, para2, para3, para4, para5
```

**With `window size = 3`, `step = 2`, chunking would result in:**

| Chunk ID | Paragraphs Included        |
| -------- | -------------------------- |
| 0        | para1, para2, para3        |
| 1        | Para2, para3, para4, para6 |
| 2        | Para4, para5               |

Each chunk **overlaps** with the previous one by 1 paragraphs. This is crucial for maintaining **contextual flow**.

------

### Why Use Sliding Window?

Ensures important context isn’t split across chunks.

Provides more robust retrieval results since overlapping content increases chances of hitting relevant context.

Especially useful when answers might span multiple paragraphs.

### Example Output:

```json
{
  "article_id": "...",
  "chunk_id": 0,
  "title": "...",
  "section": "...",
  "url": "...",
  "content": "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
}
```

📍Saved to: `RAG-Based-Vet-QA/vet_corpus/chunked_merck.jsonl`

------

## 📌 Step 3: Download Embedding Model

**Script:** `RAG-Based-Vet-QA/scripts/step3_download_bge_retriever.py`

### Purpose:

Download the sentence-transformers embedding model (`BAAI/bge-small-en-v1.5`) locally.

### Output:

Downloads to:

```bash
RAG-Based-Vet-QA/bge_retriever/
```

------

## 📌 Step 4: Build Vector Index

**Script:** `RAG-Based-Vet-QA/scripts/step4_build_vet_index.py`

### Purpose:

In this step, we use the **LlamaIndex** framework to build a local FAISS vector index from the chunked veterinary knowledge base. This index allows the system to perform efficient similarity search over the paragraph-level embeddings and retrieve the most relevant information for user queries.

### Key Components

- **Embedding model**: `BAAI/bge-small-en-v1.5`, loaded locally using `HuggingFaceEmbedding`.
- **Vector store**: FAISS index stored in the `vet_local_embedding/` directory.
- **Documents**: Input data is the paragraph-chunked file `vet_corpus/chunked_merck.jsonl`.

### Output:

Index is saved to:

```bash
RAG-Based-Vet-QA/vet_local_embedding/
```

This includes document metadata, FAISS index, and docstore.

------

## 📌 Step 5: Download Local Qwen Model

**Script:** `RAG-Based-Vet-QA/scripts/step5_download_qwen_model.py`

### Purpose:

Download and store the Qwen LLM model locally for offline usage.

```python
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
```

### Output:

Model + tokenizer saved to local directory.

```bash
RAG-Based-Vet-QA/qwen_0.5b/
```

------

## 📌 Step 6: Run QA Chatbot

**Script:** `RAG-Based-Vet-QA/scripts/step6_vet_QA_chat_1.py`

### Purpose:

This step launches an interactive RAG-based chatbot that can answer veterinary questions using information retrieved from the Merck Veterinary Manual. It combines local dense retrieval with local LLM reasoning in a conversational interface.

### Key Features:

- **Retriever**: Locally hosted embedding model `BAAI/bge-small-en-v1.5`
- **LLM**: Locally hosted `Qwen/Qwen2.5-0.5B-Instruct`, loaded with Hugging Face Transformers
- **Multi-turn memory**: Maintains conversational history for contextualized follow-up questions
- **Veterinary prompt**: Custom system prompt styled as a professional vet assistant
- **Source highlighting**: Displays deduplicated source URLs for transparency

###  Note
- You must download the local models before running this script:
  - Run `step3_download_bge_retriever.py` and `step5_download_qwen_model.py` to prepare the models in `bge_retriever/` and `qwen_0.5b/`.
- No internet or OpenAI key is required after model download — everything runs locally.
- For reproducible output, the LLM uses `temperature=0` and `do_sample=False`.

------

## Memory & Reasoning Flow

When you type a question:

1. Chunks most similar to your question (top 8) are retrieved using `BGE` vectors.
2. `Qwen` model takes those chunks + prompt + chat history as input.
3. A tailored veterinary answer is generated and sources displayed.

