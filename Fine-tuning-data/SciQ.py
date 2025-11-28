from datasets import load_dataset
import json

# 加载数据集
ds = load_dataset("allenai/sciq", cache_dir="./Fine-tuning-data/sciq_data")
print("Dataset loaded successfully.")

# 打印训练集大小
print(len(ds['train']), "training samples")

# 查看第一个样本
print("\nSample train data:", ds['train'][0])

# 输出数据结构和字段类型
def print_dataset_info(dataset_split):
    print("\nDataset structure and field types:")
    for key, value in dataset_split.features.items():
        print(f"{key}: {value}")

print_dataset_info(ds['train'])