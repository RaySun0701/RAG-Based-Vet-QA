import json
import uuid
import os

# Input and output file paths
INPUT_FILE = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_knowledge/merck_knowledge.jsonl"    # Specify your own directory
OUTPUT_FILE = "/Users/raysun/Desktop/RAG-Based-Vet-QA/vet_corpus/chunked_merck.jsonl"        # Specify your own directory

# Parameters for chunking
MAX_PARAGRAPHS = 3
STRIDE = 2

def split_with_paragraph_overlap(doc, max_paragraphs=3, stride=2):
    """
    Split a document into overlapping paragraph chunks using the 'paragraphs' field.
    """
    paragraphs = doc.get("paragraphs", [])
    chunks = []

    for i in range(0, len(paragraphs), stride):
        chunk_paragraphs = paragraphs[i:i + max_paragraphs]
        if not chunk_paragraphs:
            continue
        chunk_text = "\n".join(chunk_paragraphs)
        chunk = {
            "article_id": doc.get("article_id", str(uuid.uuid4())),
            "chunk_id": len(chunks),
            "title": doc["title"],
            "section": doc["section"],
            "url": doc["url"],
            "content": chunk_text
        }
        chunks.append(chunk)

        if i + max_paragraphs >= len(paragraphs):
            break

    return chunks

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            doc = json.loads(line)
            if "article_id" not in doc:
                doc["article_id"] = str(uuid.uuid4())  # Ensure article_id exists
            chunks = split_with_paragraph_overlap(doc, MAX_PARAGRAPHS, STRIDE)
            for chunk in chunks:
                json.dump(chunk, fout, ensure_ascii=False)
                fout.write("\n")

    print(f"Chunking complete. Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
