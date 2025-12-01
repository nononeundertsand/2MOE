import os
import json
import uuid

###############################################
# 统一格式构造函数
###############################################
def build_unified_record(
    dataset, question, passage, options, answer, answer_index, meta
):
    return {
        "id": str(uuid.uuid4()),
        "dataset": dataset,
        "question": question or "",
        "passage": passage or "",
        "options": options or [],
        "answer": answer,
        "answer_index": answer_index,
        "meta": meta or {}
    }


###############################################
# Dataset-specific Normalizers
###############################################

############## 1. AGIEval #####################
def normalize_agieval(sample, dataset_name):
    passage = sample.get("passage")
    question = sample.get("question")
    options = sample.get("options")
    label = sample.get("label")
    answer = sample.get("answer")
    other = sample.get("other")

    # options 标准化
    if options is None:
        options = []
    
    # label 可能是 str 或 list
    if isinstance(label, list):
        answer_text = label[0] if len(label) > 0 else None
    else:
        answer_text = label

    # 放入 meta
    meta = {"raw_answer": answer, "other": other}

    # answer_index
    answer_index = None
    if options and answer_text and answer_text in ["A","B","C","D"]:
        answer_index = ord(answer_text) - ord("A")

    return build_unified_record(
        dataset_name, question, passage, options, answer_text, answer_index, meta
    )


############## 2. MMLU-Pro #####################
def normalize_mmlu(sample):
    idx = sample["answer_index"]
    options = sample["options"]
    answer = options[idx]

    meta = {
        "category": sample.get("category"),
        "src": sample.get("src")
    }

    return build_unified_record(
        "MMLU-Pro",
        sample["question"],
        "",
        options,
        answer,
        idx,
        meta
    )


############## 3. SciQ #########################
def normalize_sciq(sample):
    options = [
        sample["distractor1"],
        sample["distractor2"],
        sample["distractor3"],
        sample["correct_answer"],
    ]
    answer = sample["correct_answer"]

    meta = {"support": sample.get("support")}

    return build_unified_record(
        "SciQ",
        sample["question"],
        "",
        options,
        answer,
        3,
        meta
    )


############## 4. ARC (Challenge/Easy) ##########
def normalize_arc(sample, dataset_name):
    options = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    ans_key = sample["answerKey"]

    idx = labels.index(ans_key)
    answer = options[idx]

    meta = {"id": sample["id"]}

    return build_unified_record(
        dataset_name,
        sample["question"],
        "",
        options,
        answer,
        idx,
        meta
    )


###############################################
# 自动识别数据集类型并调用对应解析函数
###############################################
def detect_and_normalize(sample, filename):
    # 1. MMLU-Pro
    if "question_id" in sample and "options" in sample and "answer_index" in sample:
        return normalize_mmlu(sample)

    # 2. SciQ
    if "correct_answer" in sample and "distractor1" in sample:
        return normalize_sciq(sample)

    # 3. ARC
    if "choices" in sample and "answerKey" in sample:
        if "challenge" in filename.lower():
            dsname = "ARC-Challenge"
        elif "easy" in filename.lower():
            dsname = "ARC-Easy"
        else:
            dsname = "ARC"
        return normalize_arc(sample, dsname)

    # 4. 默认 AGIEval
    return normalize_agieval(sample, "AGIEval")


###############################################
# 加载 JSONL 文件
###############################################
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


###############################################
# 主融合程序
###############################################
def merge_datasets(input_dir, output_file="unified_dataset.jsonl"):
    merged = []
    files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    print(f"Found {len(files)} JSONL files.")

    for fname in files:
        path = os.path.join(input_dir, fname)
        print(f"Processing: {fname}")

        samples = load_jsonl(path)

        for s in samples:
            try:
                unified = detect_and_normalize(s, fname)
                merged.append(unified)
            except Exception as e:
                print(f"[WARN] Error parsing sample in {fname}: {e}")

    # 写出 unified 数据
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in merged:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nMerged total samples: {len(merged)}")
    print(f"Saved to: {output_file}")


###############################################
# 运行
###############################################
if __name__ == "__main__":
    # TODO: 修改为你自己的数据集目录路径
    INPUT_DIR = "./datasets"  
    merge_datasets(INPUT_DIR)
