import json
import os

input_file = "./datasets/dataset_before.jsonl"
output_file = "./datasets/dataset_cleaned.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

count = 0
with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            
            # 删除字段
            data.pop("source", None)
            data.pop("task_type", None)
            data.pop("extra", None)
            
            # # 删除空的 extra
            # if "extra" in data and (data["extra"] is None or data["extra"] == {}):
            #     data.pop("extra")

            # 写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid JSON line")

print(f"清理完成！总样本数：{count}")
print(f"新文件保存路径：{output_file}")
