# from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir="./Fine-tuning-data/mmlu_pro_data")
# print("Dataset loaded successfully.")
# print(len(ds['test']), "test samples")
# print("Sample test data:", ds['test'][0])
# # 输出数据集结构和每个字段的数据类型
# def print_dataset_info(dataset_split):
#     print("\nDataset structure and field types:")
#     for key, value in dataset_split.features.items():
#         print(f"{key}: {value}")

# print_dataset_info(ds['validation'])


import json
from datasets import load_dataset

# 加载 MMLU-Pro 数据集
ds = load_dataset(
    "TIGER-Lab/MMLU-Pro",
    cache_dir="./Fine-tuning-data/mmlu_pro_data"
)

print("Dataset loaded successfully.")
print("Validation samples:", len(ds["validation"]))
print("Test samples:", len(ds["test"]))

# 输出路径
output_file = "./Fine-tuning-data/mmlu_pro_data/mmlu_pro_converted.jsonl"

# 选项字母表
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def add_option_letters(options):
    """给选项加上 (A) xxx, (B) xxx 格式标签"""
    new_options = []
    for i, opt in enumerate(options):
        letter = OPTION_LETTERS[i]
        new_options.append(f"({letter}) {opt}")
    return new_options


with open(output_file, "w", encoding="utf-8") as f_out:

    # 合并 validation + test
    for split_name in ["validation", "test"]:
        split = ds[split_name]

        for item in split:

            correct_letter = item.get("answer", "")
            answer_list = [correct_letter] if correct_letter else []

            # 给选项添加 (A), (B) ... 标签
            converted_options = add_option_letters(item.get("options", []))

            new_item = {
                "source": "MMLU-Pro",
                "task_type": "mcq",
                "passage": "",
                "question": item.get("question", ""),
                "options": converted_options,   # "(A) xxx" 格式
                "answer": answer_list,          # ["I"]
                "extra": {
                    "category": item.get("category", ""),
                    "src": item.get("src", "")
                }
            }

            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"转换完成！保存到：{output_file}")


