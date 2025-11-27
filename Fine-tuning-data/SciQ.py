from datasets import load_dataset

ds = load_dataset("allenai/sciq", cache_dir="./Fine-tuning-data/sciq_data")
print("Dataset loaded successfully.")
print(len(ds['train']), "training samples")