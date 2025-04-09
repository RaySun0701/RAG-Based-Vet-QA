from sentence_transformers import SentenceTransformer

model_id = "BAAI/bge-small-en-v1.5"
save_path = "/Users/raysun/Desktop/RAG-Based-Vet-QA/bge_retriever"

model = SentenceTransformer(model_id)
model.save(save_path)
