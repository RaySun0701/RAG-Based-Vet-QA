import argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path

def download_bge_model(save_dir: Path):
    model_id = "BAAI/bge-small-en-v1.5"
    print(f"Downloading model {model_id} ...")
    model = SentenceTransformer(model_id)
    model.save(str(save_dir))
    print(f"Model saved to {save_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BGE Retriever model")
    parser.add_argument(
        "--output_dir", type=str, default="bge_retriever",
        help="Directory to save the downloaded BGE model"
    )
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    download_bge_model(output_path)
