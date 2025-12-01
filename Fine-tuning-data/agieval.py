# from datasets import load_dataset, Dataset
# ds_agieval = load_dataset("json", data_files={"train": "./Fine-tuning-data/agieval_data/*.jsonl"})

import json
import glob

def inspect_jsonl_full_with_types(path_pattern):
    files = glob.glob(path_pattern)
    print(f"\n============================")
    print(f"Checking: {path_pattern}")
    print(f"Found {len(files)} files\n")

    for filename in files:
        print("=" * 80)
        print(f"FILE: {filename}")
        print("-" * 80)

        first_keys = None
        first_types = None
        diff_examples = []
        line_num = 0

        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"JSON parse error at line {line_num}: {e}")
                    continue

                keys = tuple(sorted(obj.keys()))
                types = {k: type(obj[k]).__name__ for k in obj.keys()}

                if first_keys is None:
                    first_keys = keys
                    first_types = types
                    print(f"Base structure (from line 1): {first_keys}")
                    print(f"Base types: {first_types}")
                else:
                    if keys != first_keys or types != first_types:
                        diff_examples.append((line_num, keys, types))

        if len(diff_examples) == 0:
            print("✓ All lines match the base structure and types.\n")
        else:
            print(f"⚠ Found {len(diff_examples)} lines with different structure or types:")
            for ln, k, t in diff_examples[:10]:  # 限制最多显示 10 条
                print(f"  Line {ln}: keys={k}, types={t}")

            if len(diff_examples) > 10:
                print(f"  ... {len(diff_examples)-10} more differences.")

        print()

# 修改为你自己的路径模式
inspect_jsonl_full_with_types("./Fine-tuning-data/agieval_data/*.jsonl")



