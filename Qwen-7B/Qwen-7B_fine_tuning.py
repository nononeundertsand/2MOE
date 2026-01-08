# train_windows_qwen7b_lora.py
"""
Single-GPU Windows-compatible training script for Qwen2.5-7B-Instruct with LoRA (PEFT).
Usage: python train_windows_qwen7b_lora.py
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import json
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model




MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 模型名称
MODEL_PATH = "./Qwen-7B/Qwen-7B-model/Qwen2.5-7B-Instruct"   # 模型路径
DATA_PATH  = "./fine_tuning_dataset/fine_tuning_dataset_oneshot.jsonl"  # 训练数据路径

OUTPUT_DIR = "./qwen2.5-7b-lora-output"  # 输出路径
CUTOFF_LEN = 8192  # 长上下文设定

# Qwen7B 显存更大 → 建议降低 batch
LR = 3e-5
EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2            # <-- 原来 8，7B 显存吃紧，改为 2（你也可以改为 1）
GRAD_ACC = 16
MAX_GRAD_NORM = 1.0

# LoRA 配置（保持高性能）
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------
# Prompt生成器，用于将一条训练数据转为prompt格式
# ------------------------------------------------------------
def build_prompt_from_example(ex: Dict) -> str:
    parts: List[str] = []
    passage = ex.get("passage")
    question = ex.get("question")
    options = ex.get("options")
    answer = ex.get("answer")

    if passage and passage.strip():
        parts.append("Context: " + passage.strip())
    if question and question.strip():
        parts.append("Question: " + question.strip())

    if options:
        parts.append("Options:")
        for o in options:
            parts.append(o.strip())

    parts.append("Answer:")
    return "\n".join(parts)


# ============================================================
# Load Dataset 加载数据集
# ============================================================
print("Loading dataset:", DATA_PATH)
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
print("Dataset loaded. Examples:", len(raw_ds))


# ============================================================
# Load tokenizer & model 加载模型和分词器
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=MODEL_PATH,
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Added pad token.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading model Qwen2.5-7B-Instruct (float32)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,    # 使用 float16 加载模型
    cache_dir=MODEL_PATH,
)

model.resize_token_embeddings(len(tokenizer))
model.to(device)

print("Model loaded.")


# ============================================================
# Apply LoRA  配置LoRA并应用到模型
# ============================================================
print("Applying LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",  # 指明任务类型，PEFT 可能针对不同任务调整内置行为。
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数的统计，
#帮助确认 LoRA 是否正确应用（会显示总参数量与可训练的参数量比例）


# ============================================================
# Tokenization 构造训练用 tokens、labels
# ============================================================
def tokenize_example(example):
    prompt = build_prompt_from_example(example)  # 构造 prompt

    answer = None
    if "answer" in example and example["answer"]:
        ans = example["answer"]
        answer = ans[0] if isinstance(ans, list) else ans

    label_text = ""
    if answer:
        if isinstance(example.get("options"), list):
            normalized = str(answer).strip()
            chosen = None
            for opt in example["options"]:
                if opt.strip().upper().startswith(normalized.upper()):
                    chosen = opt.strip()
                    break
                if opt.strip().startswith("(" + normalized.upper() + ")"):
                    chosen = opt.strip()
                    break
            if chosen:
                if chosen.startswith("(") and ")" in chosen:
                    label_text = chosen.split(")", 1)[1].strip()
                else:
                    if "." in chosen:
                        label_text = chosen.split(".", 1)[1].strip()
                    else:
                        label_text = chosen
            else:
                label_text = str(answer).strip()
        else:
            label_text = str(answer).strip()

    if label_text:
        text = prompt + " " + label_text
    else:
        text = prompt

    enc = tokenizer(text, truncation=True, max_length=CUTOFF_LEN)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.copy()

    if label_text:
        enc_prompt = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN)
        prompt_len = len(enc_prompt["input_ids"])
        for i in range(prompt_len):
            labels[i] = -100
    else:
        labels = [-100] * len(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("Tokenizing...")
tokenized = raw_ds.map(tokenize_example, remove_columns=raw_ds.column_names)
print("Tokenization completed.")


# ============================================================
# Optimizer + scheduler
# ============================================================
train_data_len = len(tokenized)
steps_per_epoch = math.ceil(train_data_len / (PER_DEVICE_BATCH_SIZE * GRAD_ACC))
total_steps = steps_per_epoch * EPOCHS

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))


# ============================================================
# Data Collator
# ============================================================
def data_collator(features: List[Dict]):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# Trainer
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    max_grad_norm=MAX_GRAD_NORM,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
)

print("Start training...")
trainer.train()
print("Training complete.")

model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved LoRA outputs to", OUTPUT_DIR)
