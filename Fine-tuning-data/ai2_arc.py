from datasets import load_dataset
import json

# 加载 ARC 数据集
ds_arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir="./Fine-tuning-data/ai2_arc_data")
ds_arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir="./Fine-tuning-data/ai2_arc_data")
print("Datasets loaded successfully.")

# 打印训练集大小
print(f"ARC Challenge train set size: {len(ds_arc_challenge['train'])}")
print(f"ARC Easy train set size: {len(ds_arc_easy['train'])}")

# 查看第一个样本
print("\nSample ARC-Challenge train data:", ds_arc_challenge['train'][0])
print("\nSample ARC-Easy train data:", ds_arc_easy['train'][0])

# 输出数据结构和字段类型
def print_dataset_info(dataset_split, name):
    print(f"\n{name} dataset structure and field types:")
    for key, value in dataset_split.features.items():
        print(f"{key}: {value}")

print_dataset_info(ds_arc_challenge['train'], "ARC-Challenge")
print_dataset_info(ds_arc_easy['train'], "ARC-Easy")

