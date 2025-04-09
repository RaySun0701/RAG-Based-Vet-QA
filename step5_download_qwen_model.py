from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen1.5-0.5B-Chat"
local_dir = "/Users/raysun/Desktop/qwen_0.5b"   # Use your own local path to save the Qwen model

AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(local_dir)
AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).save_pretrained(local_dir)
