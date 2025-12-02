from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", cache_dir="./Fine-tuning-data/mmlu_data")
print("data loaded successfully.")
print("输出数据集的总量：", len(ds['test']))
print("输出数据集的第一个样本：", ds['test'][0])