from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir="./Fine-tuning-data/mmlu_pro_data")
print("Dataset loaded successfully.")
print(len(ds['test']), "test samples")
print("Sample test data:", ds['test'][0])
# 输出数据集结构和每个字段的数据类型
def print_dataset_info(dataset_split):
    print("\nDataset structure and field types:")
    for key, value in dataset_split.features.items():
        print(f"{key}: {value}")

print_dataset_info(ds['validation'])