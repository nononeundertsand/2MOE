# train_windows_mistral7b_lora.py
"""
Single-GPU Windows-compatible training script for Mistral-7B-Instruct-v0.3 with LoRA (PEFT).
Usage: python train_windows_mistral7b_lora.py
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
from typing import Dict, List
import torch
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


# ============================================================
#                 ★★★★★ Model & Path Config ★★★★★
# ============================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"   
MODEL_PATH = "D:/2MOE/Mistral-7B/Mistral-7B-model"     # ← 修改为你的实际路径

DATA_PATH = "D:/2MOE/fine_tuning_dataset/fine_tuning_dataset_oneshot.jsonl"

OUTPUT_DIR = "./mistral7b-lora-output"
CUTOFF_LEN = 8192

# Training hyperparams
LR = 3e-5
EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACC = 16
MAX_GRAD_NORM = 1.0

# LoRA configuration
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05

# Mistral attention projection module names
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)


# ============================================================
# Prompt builder
# ============================================================
def build_prompt_from_example(ex: Dict) -> str:
    parts: List[str] = []
    passage = ex.get("passage")
    question = ex.get("question")
    options = ex.get("options")

    if passage:
        parts.append("Context: " + passage.strip())
    if question:
        parts.append("Question: " + question.strip())

    if options:
        parts.append("Options:")
        for o in options:
            parts.append(o.strip())

    parts.append("Answer:")
    return "\n".join(parts)


# ============================================================
# Load dataset
# ============================================================
print("Loading dataset...")
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
print("Dataset size:", len(raw_ds))


# ============================================================
# Load tokenizer & model
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_PATH,
)

# Mistral tokenizer uses <pad> if needed
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print("Added <pad> token to tokenizer.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading Mistral-7B-Instruct-v0.3 model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_PATH,
    torch_dtype=torch.float32,
)

model.resize_token_embeddings(len(tokenizer))
model.to(device)
print("Model loaded.")


# ============================================================
# Apply LoRA
# ============================================================
print("Applying LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# Tokenization
# ============================================================
def tokenize_example(example):
    prompt = build_prompt_from_example(example)

    # Extract label text
    answer = example.get("answer", "")
    label_text = ""

    if answer:
        if isinstance(answer, list):
            answer = answer[0]

        options = example.get("options")
        if options:
            normalized = str(answer).strip()
            chosen = None
            for opt in options:
                if opt.strip().upper().startswith(normalized.upper()):
                    chosen = opt.strip()
                    break
                if opt.strip().startswith("(" + normalized.upper() + ")"):
                    chosen = opt.strip()
                    break

            if chosen:
                # Remove "(A)" or "A." prefix
                if chosen.startswith("(") and ")" in chosen:
                    label_text = chosen.split(")", 1)[1].strip()
                elif "." in chosen:
                    label_text = chosen.split(".", 1)[1].strip()
                else:
                    label_text = chosen
            else:
                label_text = str(answer)
        else:
            label_text = str(answer)

    final_text = prompt + " " + label_text if label_text else prompt

    enc = tokenizer(final_text, truncation=True, max_length=CUTOFF_LEN)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.copy()

    # Mask prompt part
    if label_text:
        prompt_enc = tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN)
        p_len = len(prompt_enc["input_ids"])
        for i in range(p_len):
            labels[i] = -100
    else:
        labels = [-100] * len(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


print("Tokenizing dataset...")
tokenized = raw_ds.map(tokenize_example, remove_columns=raw_ds.column_names)
print("Tokenization complete.")


# ============================================================
# Optimizer + Scheduler
# ============================================================
train_data_len = len(tokenized)
steps_per_epoch = math.ceil(train_data_len / (PER_DEVICE_BATCH_SIZE * GRAD_ACC))
total_steps = steps_per_epoch * EPOCHS

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, total_steps)
)


# ============================================================
# Data Collator
# ============================================================
def data_collator(features: List[Dict]):
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

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
    save_strategy="epoch",
    fp16=False,    # 保持 fp32，避免 Windows 下可能的 bf16 不兼容
    bf16=False,
    logging_steps=50,
    report_to="none",
    max_grad_norm=MAX_GRAD_NORM,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
)

print("Starting training...")
trainer.train()
print("Training completed.")

model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved LoRA fine-tuned model to:", OUTPUT_DIR)
