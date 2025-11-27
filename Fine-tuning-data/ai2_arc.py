from datasets import load_dataset

ds_arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir="./Fine-tuning-data/ai2_arc_data")
ds_arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir="./Fine-tuning-data/ai2_arc_data")
print("Datasets loaded successfully.")
print(f"ARC Challenge train set size: {len(ds_arc_challenge['train'])}")
print(f"ARC Easy train set size: {len(ds_arc_easy['train'])}")