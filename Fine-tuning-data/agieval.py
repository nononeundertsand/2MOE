# from datasets import load_dataset, Dataset
# ds_agieval = load_dataset("json", data_files={"train": "./Fine-tuning-data/agieval_data/*.jsonl"})

# import json
# import glob

# def inspect_jsonl_full_with_types(path_pattern):
#     files = glob.glob(path_pattern)
#     print(f"\n============================")
#     print(f"Checking: {path_pattern}")
#     print(f"Found {len(files)} files\n")

#     for filename in files:
#         print("=" * 80)
#         print(f"FILE: {filename}")
#         print("-" * 80)

#         first_keys = None
#         first_types = None
#         diff_examples = []
#         line_num = 0

#         with open(filename, "r", encoding="utf-8") as f:
#             for line in f:
#                 line_num += 1
#                 try:
#                     obj = json.loads(line)
#                 except Exception as e:
#                     print(f"JSON parse error at line {line_num}: {e}")
#                     continue

#                 keys = tuple(sorted(obj.keys()))
#                 types = {k: type(obj[k]).__name__ for k in obj.keys()}

#                 if first_keys is None:
#                     first_keys = keys
#                     first_types = types
#                     print(f"Base structure (from line 1): {first_keys}")
#                     print(f"Base types: {first_types}")
#                 else:
#                     if keys != first_keys or types != first_types:
#                         diff_examples.append((line_num, keys, types))

#         if len(diff_examples) == 0:
#             print("✓ All lines match the base structure and types.\n")
#         else:
#             print(f"⚠ Found {len(diff_examples)} lines with different structure or types:")
#             for ln, k, t in diff_examples[:10]:  # 限制最多显示 10 条
#                 print(f"  Line {ln}: keys={k}, types={t}")

#             if len(diff_examples) > 10:
#                 print(f"  ... {len(diff_examples)-10} more differences.")

#         print()

# # 修改为你自己的路径模式
# inspect_jsonl_full_with_types("./Fine-tuning-data/agieval_data/*.jsonl")



import json
import os

def agieval_jec_data_merge():
    input_dir = "./Fine-tuning-data/agieval_data"
    output_dir = "./Fine-tuning-data/agieval_data_converted"
    os.makedirs(output_dir, exist_ok=True)

    input_files = [
        "jec-qa-ca.jsonl",
        "jec-qa-kd.jsonl"
    ]

    output_file = os.path.join(output_dir, "jec-qa-all_converted.jsonl")


    def convert_item(item):
        """
        将 AGIEval 的 JEC-QA 单条样本转换为统一格式。
        """
        new_item = {
            "source": "AGIEval",                  # str
            "task_type": "mcq",                   # str
            "passage": "" if item["passage"] is None else str(item["passage"]),  # str
            "question": str(item["question"]),    # str
            "options": [str(opt) for opt in item["options"]],  # list[str]
            "answer": list(item["label"]),        # list[str]
            "extra": {}
        }
        return new_item


    # 写入一个 jsonl 文件
    with open(output_file, "w", encoding="utf-8") as fout:
        for fname in input_files:
            in_path = os.path.join(input_dir, fname)

            with open(in_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    new_item = convert_item(item)
                    fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"✔ 已完成转换与合并，输出文件：{output_file}")

def agieval_sat_data_merge():
    input_dir = "./Fine-tuning-data/agieval_data"
    output_dir = "./Fine-tuning-data/agieval_data_converted"
    os.makedirs(output_dir, exist_ok=True)

    input_files = [
        "sat-en-without-passage.jsonl",
        "sat-en.jsonl",
        "sat-math.jsonl"
    ]

    output_file = os.path.join(output_dir, "sat-all_converted.jsonl")


    def convert_item(item):
        """
        将 SAT 系列样本转换为统一格式。
        """

        # passage 可能是 NoneType 或 str
        passage = item.get("passage")
        passage = "" if passage is None else str(passage)

        new_item = {
            "source": "AGIEval",
            "task_type": "mcq",
            "passage": passage,
            "question": str(item["question"]),
            "options": [str(opt) for opt in item["options"]],
            "answer": [str(item["label"])],   # label 为 str，需转为 list[str]
            "extra": item["other"] if isinstance(item.get("other"), dict) else {}
        }

        return new_item


    with open(output_file, "w", encoding="utf-8") as fout:
        for fname in input_files:
            in_path = os.path.join(input_dir, fname)

            with open(in_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    new_item = convert_item(item)
                    fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"✔ 已完成 SAT 数据集转换与合并：{output_file}")

def agieval_gaokao_data_merge():
    # 输入与输出路径
    input_dir = "./Fine-tuning-data/agieval_data"
    output_dir = "./Fine-tuning-data/agieval_data_converted"
    os.makedirs(output_dir, exist_ok=True)

    # 组 1 文件列表
    group1_files = [
        "aqua-rat.jsonl",
        "gaokao-biology.jsonl",
        "gaokao-chemistry.jsonl",
        "gaokao-chinese.jsonl",
        "gaokao-english.jsonl",
        "gaokao-geography.jsonl",
        "gaokao-history.jsonl",
        "gaokao-mathqa.jsonl",
        "gaokao-physics.jsonl",
        "logiqa-en.jsonl",
        "logiqa-zh.jsonl",
        "lsat-ar.jsonl",
        "lsat-lr.jsonl",
        "lsat-rc.jsonl"
    ]

    output_file = os.path.join(output_dir, "agieval-group1_converted.jsonl")


    def convert_item_group1(item):
        """
        转换 AGIEval 组 1 标准结构
        结构：('answer', 'label', 'options', 'other', 'passage', 'question')
        """

        # passage: None or str
        passage = item.get("passage")
        passage = "" if passage is None else str(passage)

        # label: str → list[str]
        label = item.get("label")
        answer_list = [str(label)] if label is not None else []

        # other: dict or None
        extra = item.get("other")
        extra = extra if isinstance(extra, dict) else {}

        new_item = {
            "source": "AGIEval",
            "task_type": "mcq",
            "passage": passage,
            "question": str(item["question"]),
            "options": [str(x) for x in item["options"]],
            "answer": answer_list,
            "extra": extra
        }

        return new_item


    with open(output_file, "w", encoding="utf-8") as fout:
        for fname in group1_files:
            file_path = os.path.join(input_dir, fname)

            with open(file_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    new_item = convert_item_group1(item)
                    fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print("✔ 已完成 AGIEval 组 1 的全部数据转换与合并：", output_file)




if __name__ == "__main__":
    # agieval_jec_data_merge()
    # agieval_sat_data_merge()
    agieval_gaokao_data_merge()