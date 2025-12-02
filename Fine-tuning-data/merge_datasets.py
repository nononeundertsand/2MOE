import json
import glob

# 需要合并的 6 个 jsonl 文件路径（按你的实际路径修改）
jsonl_files = [
    "./Fine-tuning-data/sciq_data/sciq_all_splits_converted.jsonl",
    "./Fine-tuning-data/mmlu_pro_data/mmlu_pro_converted.jsonl",
    "./Fine-tuning-data/ai2_arc_data/arc_all_converted_with_letters.jsonl",
     "./Fine-tuning-data/agieval_data_converted/agieval-group1_converted.jsonl",
    "./Fine-tuning-data/agieval_data_converted/jec-qa-all_converted.jsonl",
     "./Fine-tuning-data/agieval_data_converted/sat-all_converted.jsonl",
]

# 输出文件
output_file = "./datasets/dataset.jsonl"

# 确保输出目录存在
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)

total_count = 0

with open(output_file, "w", encoding="utf-8") as fout:
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                # 可选：验证 json 格式合法
                try:
                    json_obj = json.loads(line)
                    fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                    total_count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file_path}")

print(f"合并完成！总样本数：{total_count}")
print(f"最终 dataset.jsonl 保存路径：{output_file}")
