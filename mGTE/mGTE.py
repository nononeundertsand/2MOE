# # Load model directly
# from transformers import AutoModel

# model = AutoModel.from_pretrained(
#     "Alibaba-NLP/gte-multilingual-base",
#     trust_remote_code=True,
#     # dtype="auto",
#     cache_dir="D:/2MOE/mGTE/mGTE_model"  # 这里指定路径
# )


import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import os

# 临时指定 loky 使用的 CPU 核心数
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 之后再 import KMeans 或其他依赖 joblib 的库
from sklearn.cluster import KMeans

# -------------------------------
# Step 0: 配置
# -------------------------------
DATA_PATH = "D:/2MOE/datasets/dataset.jsonl"  # 原始 JSONL 数据集
OUTPUT_DIR = "D:/2MOE/fine_tuning_dataset"  # 输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32  # 批量处理 embedding
NUM_CLUSTERS = 8  # KMeans 聚类数
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
CACHE_DIR = "D:/2MOE/mGTE/mGTE_model"

# -------------------------------
# Step 1: 加载模型和 tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -------------------------------
# Step 2: 读取 JSONL 数据集
# -------------------------------
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data.append(item)

print(f"共读取 {len(data)} 条题目")

# -------------------------------
# Step 3: 批量生成 embedding（question + passage）
# -------------------------------
def embed_texts(data_list, batch_size=32):
    """
    data_list: 原始数据列表，每条至少包含 'question' 和 'passage'
    """
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data_list), batch_size), desc="生成 embedding"):
            batch_data = data_list[i:i+batch_size]

            # 拼接 question + passage
            batch_texts = [
                item["question"] + " " + item.get("passage", "")
                for item in batch_data
            ]

            # Tokenize
            tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

            # 前向推理
            outputs = model(**tokens)

            # 使用 [CLS] token 作为句子向量
            batch_emb = outputs.last_hidden_state[:, 0, :].cpu()

            # 归一化向量
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            embeddings.append(batch_emb)

    return torch.cat(embeddings, dim=0)


# 生成 embedding
embeddings = embed_texts(data, batch_size=BATCH_SIZE)
print("Embedding shape:", embeddings.shape)

# -------------------------------
# Step 4: K-Means 聚类
# -------------------------------
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
clusters = kmeans.fit_predict(embeddings.numpy())
print("聚类分布:", {i: sum(clusters == i) for i in range(NUM_CLUSTERS)})

# -------------------------------
# Step 5: 保存聚类结果
# -------------------------------
for i in range(NUM_CLUSTERS):
    cluster_data = [data[j] for j in range(len(data)) if clusters[j] == i]
    out_file = os.path.join(OUTPUT_DIR, f"fine_tuning_dayaset_{i}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in cluster_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"{NUM_CLUSTERS} 个聚类数据集已保存到 {OUTPUT_DIR}")
