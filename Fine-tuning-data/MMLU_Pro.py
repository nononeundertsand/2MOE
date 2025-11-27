from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir="./Fine-tuning-data/mmlu_pro_data")
print("Dataset loaded successfully.")
print(len(ds['test']), "test samples")