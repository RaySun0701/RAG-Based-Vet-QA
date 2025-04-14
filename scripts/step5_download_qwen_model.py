from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define model name and target path
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Save to ./qwen_0.5b relative to project root
LOCAL_DIR = Path(__file__).resolve().parent.parent / "qwen_0.5b"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Download and save tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(LOCAL_DIR)

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained(LOCAL_DIR)

print(f"Model downloaded and saved to: {LOCAL_DIR}")
