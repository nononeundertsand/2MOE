import os
import math
from typing import List, Dict
import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ================= 配置对齐论文 Table V =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = "./Qwen-7B/Qwen-7B-model/Qwen2.5-7B-Instruct"
DATA_PATH  = "./fine_tuning_dataset/fine_tuning_dataset_oneshot.jsonl"
OUTPUT_DIR = "./qwen2.5-7b-lora-moe2"

# 论文核心参数
CUTOFF_LEN = 8192
LR = 5e-5             # 论文 Table V 设定
EPOCHS = 3            # 论文 Table V 设定
PER_DEVICE_BATCH_SIZE = 4 # 双卡共 8，对齐论文 Batch Size 8
GRAD_ACC = 16         # 论文 Table V 设定
MAX_GRAD_NORM = 1.0   # 论文 Table V 设定

# LoRA 核心参数 (对齐论文)
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

set_seed(42)

# ================= 数据处理 =================
def build_prompt_from_example(ex: Dict) -> str:
    parts = []
    if ex.get("passage"): parts.append(f"Context: {ex['passage'].strip()}")
    if ex.get("question"): parts.append(f"Question: {ex['question'].strip()}")
    if ex.get("options"):
        parts.append("Options:")
        parts.extend([o.strip() for o in ex["options"]])
    parts.append("Answer:")
    return "\n".join(parts)

def tokenize_example(example):
    prompt = build_prompt_from_example(example)
    answer = example.get("answer", "")
    if isinstance(answer, list): answer = answer[0]
    
    full_text = f"{prompt} {answer}"
    enc = tokenizer(full_text, truncation=True, max_length=CUTOFF_LEN, padding=False)
    
    # 计算 Prompt 长度以便在 Label 中 Mask 掉
    prompt_enc = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN, padding=False)
    prompt_len = len(prompt_enc["input_ids"])
    
    labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }

# ================= 模型加载 =================
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 双卡 Linux 服务器建议使用 bf16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    # device_map 在分布式训练时由 Trainer 处理，不在此手动指定
)

# 启用梯度检查点以节省显存 (处理 8192 长度必开)
model.gradient_checkpointing_enable()

# LoRA 配置
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= 训练准备 =================
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
tokenized_ds = raw_ds.map(tokenize_example, remove_columns=raw_ds.column_names)

# 优化器与调度器对齐论文 (CosineAnnealing)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine", # 对齐 Table V: CosineAnnealingLR
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                  # A800/RTX3090+ 使用 bf16 性能更好
    tf32=True,                  # 开启 Ampere 显卡加速
    max_grad_norm=MAX_GRAD_NORM,
    gradient_checkpointing=True,
    report_to="none",
    ddp_find_unused_parameters=False, # 提升分布式效率
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
)

print("Start Training on Multiple GPUs...")
trainer.train()

# 保存
if trainer.is_world_process_zero():
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Master process saved model to {OUTPUT_DIR}")