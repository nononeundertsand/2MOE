import os
# 建议在 Linux 环境下设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# ============================================================
#                   配置区
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = "./Qwen-7B/Qwen-7B-model/Qwen2.5-7B-Instruct"   
DATA_PATH  = "./fine_tuning_dataset/fine_tuning_dataset_oneshot.jsonl"

OUTPUT_DIR = "./qwen2.5-7b-lora-output"
CUTOFF_LEN = 8192 

LR = 3e-5
EPOCHS = 3
# 注意：多卡训练时，这里的 batch 是“每张卡”的 batch。
# 双卡总 batch = 2 (卡) * PER_DEVICE_BATCH_SIZE * GRAD_ACC
PER_DEVICE_BATCH_SIZE = 2            
GRAD_ACC = 8 # 既然有两张卡，可以适当降低累积步数加快速度
MAX_GRAD_NORM = 1.0

LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

SEED = 42
torch.manual_seed(SEED)

# [修改点 1] 移除所有手动 model.to(device) 的逻辑，交给 Trainer 自动处理

# ------------------------------------------------------------
# Prompt 生成器 (保持不变)
# ------------------------------------------------------------
def build_prompt_from_example(ex: Dict) -> str:
    parts: List[str] = []
    passage = ex.get("passage")
    question = ex.get("question")
    options = ex.get("options")
    if passage: parts.append("Context: " + passage.strip())
    if question: parts.append("Question: " + question.strip())
    if options:
        parts.append("Options:")
        for o in options: parts.append(o.strip())
    parts.append("Answer:")
    return "\n".join(parts)

# ============================================================
# 加载数据 (保持不变)
# ============================================================
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

# ============================================================
# [修改点 2] 加载模型与分词器 (针对 4090 优化)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

print("Loading model in BF16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,    # [修改] 4090 必须用 bfloat16，速度快且稳
    cache_dir=MODEL_PATH,
    # [修改] 如果安装了 flash-attn，建议开启：
    # attn_implementation="flash_attention_2", 
)
model.resize_token_embeddings(len(tokenizer))

# ============================================================
# [修改点 3] LoRA 配置
# ============================================================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# [重要] 开启梯度检查点，防止 7B 模型在 24G 显存上 OOM
model.gradient_checkpointing_enable() 

# ============================================================
# Tokenization (保持不变)
# ============================================================
def tokenize_example(example):
    prompt = build_prompt_from_example(example)
    answer = example.get("answer")
    label_text = str(answer[0] if isinstance(answer, list) else answer).strip()
    
    text = prompt + " " + label_text
    enc = tokenizer(text, truncation=True, max_length=CUTOFF_LEN)
    input_ids = enc["input_ids"]
    labels = input_ids.copy()
    
    enc_prompt = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN)
    prompt_len = len(enc_prompt["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    
    return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "labels": labels}

tokenized = raw_ds.map(tokenize_example, remove_columns=raw_ds.column_names)

# ============================================================
# [修改点 4] 简化 Trainer 配置 (移除手动 Optimizer)
# ============================================================
def data_collator(features: List[Dict]):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine", # 在这里配置 scheduler
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    # --- 多卡与 4090 关键配置 ---
    bf16=True,                   # 开启 BF16 混合精度
    tf32=True,                   # 开启 4090 的 TF32 算力加速
    gradient_checkpointing=True, # 进一步节省显存
    ddp_find_unused_parameters=False, # 提升分布式效率
    dataloader_num_workers=4,    # Linux 下多线程加载数据
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Start training...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)