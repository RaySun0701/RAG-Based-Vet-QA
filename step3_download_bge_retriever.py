from sentence_transformers import SentenceTransformer

model_id = "BAAI/bge-small-en-v1.5"
save_path = "/Users/raysun/Desktop/bge_retriever"  # Use your own local path to save the BGE model

model = SentenceTransformer(model_id)
model.save(save_path)
