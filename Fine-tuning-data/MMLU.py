from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", cache_dir="./Fine-tuning-data/mmlu_data")
print("data loaded successfully.")