# from datasets import load_dataset
# import json

# # 加载数据集
# ds = load_dataset("allenai/sciq", cache_dir="./Fine-tuning-data/sciq_data")
# print("Dataset loaded successfully.")

# # 打印训练集大小
# print(len(ds['train']), "training samples")

# # 查看第一个样本
# print("\nSample train data:", ds['train'][0])

# # 输出数据结构和字段类型
# def print_dataset_info(dataset_split):
#     print("\nDataset structure and field types:")
#     for key, value in dataset_split.features.items():
#         print(f"{key}: {value}")

# print_dataset_info(ds['train'])



import json
from datasets import load_dataset

# 加载 SciQ 数据集
ds = load_dataset("allenai/sciq", cache_dir="./Fine-tuning-data/sciq_data")
print("Dataset loaded successfully.")
print("Train:", len(ds["train"]), "Validation:", len(ds["validation"]), "Test:", len(ds["test"]))

output_path = "./Fine-tuning-data/sciq_data/sciq_all_splits_converted.jsonl"

OPTION_LETTERS = "ABCD"

def build_mcq_options(correct, d1, d2, d3):
    """
    构造四选一 MCQ：
    正确答案始终为 (A)，三个错误项 B/C/D
    """
    options_raw = [correct, d1, d2, d3]
    options = [f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options_raw)]
    return options, ["A"]


splits = ["train", "validation", "test"]

with open(output_path, "w", encoding="utf-8") as f_out:
    for split in splits:
        for item in ds[split]:
            correct = item.get("correct_answer", "").strip()
            d1 = item.get("distractor1", "").strip()
            d2 = item.get("distractor2", "").strip()
            d3 = item.get("distractor3", "").strip()
            support = item.get("support", "")
            question = item.get("question", "")

            # 构造选项 & 答案
            options, answer_list = build_mcq_options(correct, d1, d2, d3)

            new_item = {
                "source": "SCIQ",
                "task_type": "mcq",
                "passage": support,     # support 当成 passage
                "question": question,
                "options": options,
                "answer": answer_list,
                "extra": {}
            }

            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"SCIQ 全部 splits（train+validation+test）已转换完成！保存路径：{output_path}")

