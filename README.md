

## 🐾 Veterinary QA RAG System – Full Project Report

### ✅ Project Goal

Build a **local Retrieval-Augmented Generation (RAG)** pipeline that answers veterinary questions by retrieving and generating responses based on the **Merck Veterinary Manual** content using **local models**:

- Local embedding model: `BAAI/bge-small-en-v1.5`
- Local LLM: `Qwen/Qwen1.5-0.5B-Chat`

------

### 📁 Step-by-Step Pipeline

------

## **Step 1: Crawl Merck Veterinary Manual**

**Script:** `step1_vet_merck_scraper.py`

### 📌 Purpose:

Scrape all valid articles from the Merck Veterinary Manual website into a structured format.

### 📌 Key Features:

- Skips non-article pages (e.g., `/authors/`, `/videos/`)
- Parses:
  - `title`
  - `section`
  - `url`
  - `paragraphs`: extracted as a **list of paragraphs** (not raw text)

### 📌 Example Output:

```json
{
  "title": "Description and Physical Characteristics of Amphibians",
  "section": "Veterinary > All Other Pets > Amphibians > Description and Physical Characteristics of Amphibians",
  "url": "https://www.merckvetmanual.com/all-other-pets/amphibians/description-and-physical-characteristics-of-amphibians",
  "paragraphs": ["...", "...", "..."]
}
```

📍Saved to: `merck_knowledge.jsonl`

------

## **Step 2: Paragraph-based Chunking**

**Script:** `step2_chunk_merck_by_paragraph.py`

### 📌 Purpose:

Convert each article’s paragraph list into **overlapping chunks** for better context retention during retrieval.

### 📌 Technique:

- Sliding window:
  - Window size: 4
  - Step: 2
- Each chunk includes:
  - `chunk_id`
  - `article_id`
  - metadata (title, section, url)
  - `content`: merged paragraph text

Assume a document has the following paragraphs:

```text
para1, para2, para3, para4, para5, para6, para7
```

### With `window size = 4`, `step = 2`, chunking would result in:

| Chunk ID | Paragraphs Included        |
| -------- | -------------------------- |
| 0        | para1, para2, para3, para4 |
| 1        | para3, para4, para5, para6 |
| 2        | para5, para6, para7        |

Each chunk **overlaps** with the previous one by 2 paragraphs. This is crucial for maintaining **contextual flow**.

------

## Why Use Sliding Window?

✅ Ensures important context isn’t split across chunks.

✅ Provides more robust retrieval results since overlapping content increases chances of hitting relevant context.

✅ Especially useful when answers might span multiple paragraphs.

### 📌 Example Output:

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

📍Saved to: `chunked_merck.jsonl`

------

## **Step 3: Download Embedding Model**

**Script:** `step3_download_bge_retriever.py`

### 📌 Purpose:

Download the sentence-transformers embedding model (`BAAI/bge-small-en-v1.5`) locally.

### 📌 Output:

Downloads to:

```bash
/Users/.../.../bge_retriever/    # Use your own local path to save the BGE model
```

------

## **Step 4: Build Vector Index**

**Script:** `step4_build_vet_index.py`

### 📌 Purpose:

Embed all chunks using the **local embedding model** and build a **FAISS-based vector index** using LlamaIndex.

### 📌 Features:

- Progress bar (`tqdm`)
- Uses `HuggingFaceEmbedding` with path to local `bge_retriever`
- LLM is **not required** for this step

### 📌 Output:

Index is saved to:

```bash
/Users/.../.../vet_local_embedding/
```

------

## **Step 5: Download Local Qwen Model**

**Script:** `step5_download_qwen_model.py`

### 📌 Purpose:

Download and store the Qwen LLM model locally for offline usage.

```python
model_id = "Qwen/Qwen1.5-0.5B-Chat"
local_dir = "/Users/.../.../qwen_0.5b"   # Use your own local path to save the Qwen model
```

### 📌 Output:

Model + tokenizer saved to local directory.

------

## **Step 6: Run QA Chatbot**

**Script:** `step6_vet_QA_chat_1.py`

### 📌 Purpose:

Launch interactive terminal chatbot for question-answering using:

- Local vector index
- Local `Qwen` model as LLM
- Local `BGE` model for embedding
- Multi-turn conversation memory
- Veterinary system prompt

### 🛠 Key Config:

- Prompt: *“You are a professional veterinary assistant...”*

- Deduplicated source display

- Deterministic generation:

  ```python
  "do_sample": False,
  "top_p": None
  ```

------

## 🧪 Example Questions to Try

```text
1. What are the common causes of vomiting in dogs?
2. How should I feed a pet amphibian?
3. What are the symptoms of kidney disease in cats?
4. Can I give my dog human medication?
5. What is the lifespan of a typical rabbit?
6. How do I treat ear infections in dogs?
```

------

### 🗃 File/Directory Overview

| Path                    | Description                     |
| ----------------------- | ------------------------------- |
| `merck_knowledge.jsonl` | Raw scraped articles            |
| `chunked_merck.jsonl`   | Paragraph-based document chunks |
| `/qwen_0.5b`            | Local Qwen model                |
| `/bge_retriever`        | Local embedding model           |
| `/vet_local_embedding`  | FAISS index directory           |

------

## 🧠 Memory & Reasoning Flow

When you type a question:

1. Chunks most similar to your question (top 8) are retrieved using `BGE` vectors.
2. `Qwen` model takes those chunks + prompt + chat history as input.
3. A tailored veterinary answer is generated and sources displayed.

