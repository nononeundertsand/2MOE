# from datasets import load_dataset
# import json

# # 加载 ARC 数据集
# ds_arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir="./Fine-tuning-data/ai2_arc_data")
# ds_arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir="./Fine-tuning-data/ai2_arc_data")
# print("Datasets loaded successfully.")

# # 打印训练集大小
# print(f"ARC Challenge train set size: {len(ds_arc_challenge['train'])}")
# print(f"ARC Easy train set size: {len(ds_arc_easy['train'])}")

# # 查看第一个样本
# print("\nSample ARC-Challenge train data:", ds_arc_challenge['train'][0])
# print("\nSample ARC-Easy train data:", ds_arc_easy['train'][0])

# # 输出数据结构和字段类型
# def print_dataset_info(dataset_split, name):
#     print(f"\n{name} dataset structure and field types:")
#     for key, value in dataset_split.features.items():
#         print(f"{key}: {value}")

# print_dataset_info(ds_arc_challenge['train'], "ARC-Challenge")
# print_dataset_info(ds_arc_easy['train'], "ARC-Easy")


import json
from datasets import load_dataset

# 加载 ARC 数据集
ds_arc_challenge = load_dataset(
    "allenai/ai2_arc", "ARC-Challenge", cache_dir="./Fine-tuning-data/ai2_arc_data"
)
ds_arc_easy = load_dataset(
    "allenai/ai2_arc", "ARC-Easy", cache_dir="./Fine-tuning-data/ai2_arc_data"
)

print("Datasets loaded successfully.")

output_file = "./Fine-tuning-data/ai2_arc_data/arc_all_converted_with_letters.jsonl"

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
splits = ["train", "validation", "test"]

def add_option_letters(options):
    """给选项加上 (A) xxx, (B) xxx 格式标签"""
    new_options = []
    for i, opt in enumerate(options):
        letter = OPTION_LETTERS[i]
        new_options.append(f"({letter}) {opt}")
    return new_options

def convert_arc_item(item, source_name):
    """
    将 ARC 原始样本转换为统一 mcq json 格式，选项带 (A)(B)...
    """
    options = add_option_letters(item["choices"]["text"])
    answer = [item["answerKey"]]  # 保持原 answerKey，如 "A", "B", ...

    new_item = {
        "source": "AI2-ARC",
        "task_type": "mcq",
        "passage": "",  # ARC 数据没有 passage
        "question": item["question"],
        "options": options,
        "answer": answer,
        "extra": {"other": {"source": source_name}}
    }
    return new_item

with open(output_file, "w", encoding="utf-8") as f_out:

    # 遍历 ARC-Challenge
    for split in splits:
        for item in ds_arc_challenge[split]:
            new_item = convert_arc_item(item, "ARC-Challenge")
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    # 遍历 ARC-Easy
    for split in splits:
        for item in ds_arc_easy[split]:
            new_item = convert_arc_item(item, "ARC-Easy")
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"ARC 数据集（Easy + Challenge + 全 splits）转换完成，带选项字母标签，保存到：{output_file}")

