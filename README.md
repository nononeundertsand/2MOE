# MoE² 论文复现全流程说明文档

本文档用于指导完整复现 MoE² 系统，包括数据准备、聚类、专家构建、量化、Gating 网络训练、MoE 推理、SMO 子集选择、以及系统级延迟/能耗评测。

---

# Overview

MoE² 是一个面向异构边缘设备的多专家（Mixture-of-Experts）LLM 推理系统。
论文实验包含：

* **8 个领域专家模型**（来自 Qwen2.5-3B / Qwen2.5-7B）
* **一个 gating 网络**（选择 experts）
* **一个 SMO（Strongest Minority Oracle）专家子集选择器**
* **真实设备测试：RTX 4090 + Jetson Orin**
* **系统级指标：延迟 / 能耗 / 准确率**

本 README 是整个复现项目的总规划文档。

---

# 复现流程总览

MoE² 的复现分为 **5 个主要阶段 / 14 个步骤**：

```
Phase 0：准备与理解论文
Phase 1：构建 8 个领域专家模型（FT0~FT7）
Phase 2：训练 gating 网络
Phase 3：实现 MoE² + SMO 推理系统
Phase 4：系统层延迟 & 能耗评测
Phase 5：最终结果整理与复现验证
```


---

# Phase 0 — 环境与准备工作

### **Step 0.1：配置运行环境**

需要安装的框架：

```
python >= 3.10
pytorch >= 2.1
transformers >= 4.40
peft（LoRA）
sentence-transformers（embedding）
scikit-learn（聚类）
bitsandbytes（4-bit 量化）
```

可选（真实 testbed）：

* Jetson Orin × 6
* RTX 4090 × 3

---

# Phase 1 — 构建 8 个领域专家（Domain Experts）

在这一阶段需要逐步构建 8 个专家模型，为后续的分布式部署训练做准备。

---

## **Step 1.1：准备训练数据**

论文使用的训练数据集：

* **MMLU-Pro**
* **ARC**
* **SciQ**
* **AGIEval**

| 数据集 | 主要用途 | 数据量/规模 | 特点 | 下载源 |
|--------|----------|--------------|--------|----------|
| **MMLU-Pro** | 综合知识 + 多领域推理评测 | 12,102 题；约 4.19 MB | 比 MMLU 更难、更稳健；10 选项减少猜测影响；覆盖 14 个领域 | GitHub: https://github.com/TIGER-AI-Lab/MMLU-Pro<br>HF: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro |
| **ARC** | 小学～中学科学知识 + 推理评测 | 7,787 题；包含 14M 语料库（ARC Corpus） | 由 Easy + Challenge 组成；Challenge 难度高；测试科学理解而非记忆 | 论文: https://arxiv.org/abs/1803.05457 <br>HF: https://huggingface.co/datasets/allenai/ai2_arc|
| **SciQ** | 科学（物理/化学/生物）多选推理评测 | 13,679 题；数据约 2.82MB（下载）/10.5MB（展开） | 4 选项；提供 supporting paragraph（支持文本）；适合训练和评估 | HF: https://huggingface.co/datasets/allenai/sciq |
| **AGIEval** | 人类考试（SAT/高考/资格考试）能力评测 | 20 个任务；MCQ + Cloze | 来源为真实高标准考试；覆盖语文/数学/法律等多领域 | GitHub: https://github.com/ruixiangcui/AGIEval<br>Paper: https://arxiv.org/abs/2304.06364 |


将它们全部合并成统一格式：

```
data/combined_train.jsonl
```

---

## **Step 1.2：生成聚类用的 embedding**

论文使用 mGTE encoder，你可以使用替代方案（如 sentence-transformers）。
mGTE的模型链接：https://huggingface.co/Alibaba-NLP/gte-multilingual-base

输出：

```
embeddings.npy    # 每条数据的向量表示
```

---

## **Step 1.3：执行 K-means 聚类（K=8）**

将训练数据聚成 8 类，每类对应一个 expert：

```
cluster_0.jsonl
cluster_1.jsonl
...
cluster_7.jsonl
```

---

## **Step 1.4：对每个 cluster 进行 LoRA Fine-tune**

论文中 base-model 选择：

| 专家编号    | Base LLM            |
| ------- | ------------------- |
| 0, 1, 2 | Qwen2.5-7B-Instruct |
| 3 – 7   | Qwen2.5-3B-Instruct |

训练超参（来自论文 Table V）：

* LoRA rank = 256
* LoRA α = 512
* dropout = 0.05
* cutoff length = 8192
* epochs = 3
* learning rate = 5e-5

得到8个微调后的专家模型：

```
experts/Qwen2.5-7B-FT0
experts/Qwen2.5-7B-FT1
...
experts/Qwen2.5-3B-FT7
```
原文中提到用 8 × A800 来完成全部的微调过程（等效A100的算力）

---

## **Step 1.5：量化所有专家模型到 4-bit**

论文全部使用 Q4 量化，使用 GPTQ/AWQ/bitsandbytes 均可：

输出：

```
experts/Qwen2.5-7B-FT0-Q4
experts/Qwen2.5-7B-FT1-Q4
...
experts/Qwen2.5-3B-FT7-Q4
```

---

# Phase 2 — 训练 Gating 网络

Gating 网络根据输入 prompt 选择最合适的一组 experts。

---

## **Step 2.1：准备 gating 训练集**

论文使用：

* MMLU 的 **80%** → gating 训练
* MMLU 的 **20%** → 最终测试

输出：

```
gating_train.jsonl
gating_test.jsonl
```

---

## **Step 2.2：训练 Gating 网络（两层 MLP）**

输入：prompt 的 embedding
输出：8 维 logits（对应 8 个专家）

输出：

```
gating/gating.pt
```

---

# Phase 3 — 实现 MoE² 推理框架 + SMO

---

## **Step 3.1：实现专家远程调用（RPC）/ 本地模拟**

如果你没有真实多设备环境，可以使用 multiprocessing 模拟。

输出：

```
moe/expert_client.py
```

---

## **Step 3.2：实现 MoE 前向推理（Forward）**

包括：

1. gating 选择 top-k 专家
2. 并行向专家发送请求
3. 收集专家回答
4. 进行 majority voting / ensemble

输出：

```
moe/moe_forward.py
```

---

## **Step 3.3：实现 SMO 专家子集选择算法**

SMO 目标：

[
S^* = \arg\max Acc(S)
\quad
\text{s.t. } τ(S) ≤ τ_{max},; E(S) ≤ E_{max}
]

* τ：系统延迟
* E：能耗

因为专家数目是 8，所以子集总数是 256，穷举即可。

输出：

```
moe/smo.py
```

---

# Phase 4 — 系统级延迟 & 能耗评测

---

## **Step 4.1：测量每个专家的 latency / energy 曲线**

需要测试不同 prompt 长度下：

* 推理延迟
* 推理能耗

输出：

```
latency.json
energy.json
```

对应论文中的：

* Figure 4（延迟）
* Figure 5（能耗）

---

## **Step 4.2：在不同约束下运行 SMO**

论文测试的约束包括：

**延迟约束：**

```
1s, 2s, 3s
```

**能耗约束：**

```
5J, 10J, 15J, ..., 50J
```

对每个约束运行 SMO，得到在 MMLU 上的最终精度。

输出对应论文的：

* Table II
* Table III
* Table IV

---

# Phase 5 — 最终结果整理

请将所有实验输出整理到：

```
results/
   accuracy/
   latency/
   energy/
   smo_tables/
   figures/
```

并验证是否与论文结果一致。

---

# 推荐的项目目录结构

```
moe2/
├── data/
├── experts/
├── gating/
├── moe/
├── results/
├── scripts/
└── README.md
```


