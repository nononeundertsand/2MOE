# train_windows.py
"""
Single-GPU Windows-compatible training script for Qwen2.5-3B-Instruct with LoRA (PEFT).
Usage: python train_windows.py
Requirements:
    pip install transformers datasets accelerate peft bitsandbytes torch --upgrade
Note: bitsandbytes isn't used here but commonly helpful for quantized setups. This script uses full precision (fp32).
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import json
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# -------------------------
# User config (edit if needed)
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH = "./2MOE/Qwen-3B/Qwen_model/Qwen2.5-3B-Instruct"
DATA_PATH = "./2MOE/fine_tuning_dataset/fine_tuning_dataset_oneshot.jsonl"   # <-- change if needed
OUTPUT_DIR = "./qwen2.5-3b-lora-output"
CUTOFF_LEN = 8192

# Hyperparameters (from your spec)
LR = 5e-5
EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
MAX_GRAD_NORM = 1.0

# LoRA params
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # common targets; adjust if model differs

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)

# -------------------------
# Utility: build prompt from json example
# Handles examples that have only passage, only question, or both
# -------------------------
def build_prompt_from_example(ex: Dict) -> str:
    parts: List[str] = []
    passage = ex.get("passage")
    question = ex.get("question")
    options = ex.get("options")  # list or None
    answer = ex.get("answer")    # list or str or None

    if passage and passage.strip():
        parts.append("Context: " + passage.strip())
    if question and question.strip():
        parts.append("Question: " + question.strip())

    if options:
        # options may be like ["(A) ...", ...] or ["A. ..."]
        parts.append("Options:")
        for o in options:
            parts.append(o.strip())

    # We will add "Answer:" as the model should generate the answer token(s).
    parts.append("Answer:")

    prompt = "\n".join(parts)
    # Optionally append the reference answer as label (we handle in tokenization labeling)
    return prompt

# -------------------------
# Load dataset and transform
# -------------------------
print("Loading dataset:", DATA_PATH)
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")

print("Dataset loaded. Examples:", len(raw_ds))

# -------------------------
# Tokenizer & Model (full precision)
# -------------------------
print("Loading tokenizer and model (full precision fp32). This may take a while...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_PATH)
# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Added pad token to tokenizer.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    cache_dir=MODEL_PATH,
)
model.resize_token_embeddings(len(tokenizer))  # in case we added pad token
model.to(device)
print("Model loaded to device.")

# -------------------------
# Apply LoRA (PEFT)
# -------------------------
print("Applying LoRA (PEFT) with r=%d alpha=%d dropout=%g" % (LORA_R, LORA_ALPHA, LORA_DROPOUT))
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

# -------------------------
# Preprocessing / tokenization
# -------------------------
def tokenize_example(example):
    prompt = build_prompt_from_example(example)
    # If dataset contains a ground-truth answer, we'll make labels so model learns to generate that
    answer = None
    if "answer" in example and example["answer"]:
        # answer may be list like ["A"] or ["(A) ..."] depending; try to use first element intelligently
        ans = example["answer"]
        if isinstance(ans, list) and len(ans) >= 1:
            answer = ans[0]
        elif isinstance(ans, str):
            answer = ans
    # If answer is a single letter like "A", convert to the option text if possible
    label_text = ""
    if answer:
        # try to map "A" -> the corresponding option text (strip label markers)
        if isinstance(example.get("options"), list):
            # match by starting letter
            normalized = str(answer).strip()
            chosen = None
            # common option formats: "(A) text", "A. text", "A) text"
            for opt in example["options"]:
                # check if opt begins with A / (A) / A.
                if opt.strip().upper().startswith(normalized.upper()):
                    chosen = opt.strip()
                    break
                # also try "(A)" form
                if opt.strip().startswith("(" + normalized.upper() + ")"):
                    chosen = opt.strip()
                    break
            if chosen is None:
                # fallback: just use the answer token itself
                label_text = str(answer).strip()
            else:
                # remove leading label like "(A) " or "A. "
                # find first space after ) or .
                # but keep the whole option if it's fine
                # we'll set label_text to the chosen option with the label removed if possible
                # try to split after a closing parenthesis or dot
                if chosen.startswith("(") and ")" in chosen:
                    label_text = chosen.split(")", 1)[1].strip()
                else:
                    # split on first dot or parenthesis if present
                    if "." in chosen and chosen.index(".") <= 3:
                        label_text = chosen.split(".", 1)[1].strip()
                    elif ")" in chosen and chosen.index(")") <= 3:
                        label_text = chosen.split(")", 1)[1].strip()
                    else:
                        # fallback to whole
                        label_text = chosen
        else:
            label_text = str(answer).strip()

    # Build final text for tokenization: prompt + " " + label_text (label_text may be empty)
    # We keep the model target to generate the label_text tokens after the prompt.
    if label_text:
        # we want to compute labels that only penalize generated tokens; so we'll create input_ids for (prompt + label)
        text = prompt + " " + label_text
    else:
        # if no label, we just let model learn to produce an empty string? we'll still train on prompt only (unsupervised)
        text = prompt

    # tokenization
    enc = tokenizer(
        text,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    # compute label mask: we want labels = -100 for prompt tokens, and actual ids for label tokens (if label exists)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.copy()
    if label_text:
        # determine how many tokens in prompt alone
        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
        )
        prompt_len = len(enc_prompt["input_ids"])
        # set label tokens for prompt positions to -100
        for i in range(prompt_len):
            if i < len(labels):
                labels[i] = -100
    else:
        # no label -> set all labels to -100 so loss won't be computed (optionally you can set causal LM target)
        labels = [-100] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

print("Tokenizing dataset (this may take time)...")
tokenized = raw_ds.map(tokenize_example, remove_columns=raw_ds.column_names, num_proc=1)
print("Tokenization done. Tokenized examples:", len(tokenized))

# -------------------------
# DataLoader via Trainer expects a Dataset-like object
# -------------------------
# Compute total training steps for scheduler T_max
train_data_len = len(tokenized)
effective_batch = PER_DEVICE_BATCH_SIZE * 1  # single GPU
steps_per_epoch = math.ceil(train_data_len / (effective_batch * GRADIENT_ACCUMULATION_STEPS))
total_training_steps = steps_per_epoch * EPOCHS
print(f"Train size: {train_data_len}, steps_per_epoch: {steps_per_epoch}, total_steps: {total_training_steps}")

# -------------------------
# Create optimizer and scheduler (AdamW + CosineAnnealingLR)
# -------------------------
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
# CosineAnnealingLR requires T_max; set it to total_training_steps (at least 1)
T_max = max(1, total_training_steps)
scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

# -------------------------
# TrainingArguments for Trainer
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_strategy="epoch",
    fp16=False,            # full precision as requested
    bf16=False,
    remove_unused_columns=False,
    learning_rate=LR,
    report_to="none",
    max_grad_norm=MAX_GRAD_NORM,
)

# -------------------------
# Data collator: pad to longest in batch, keep labels as provided
# -------------------------
def data_collator(features: List[Dict]):
    # features contain input_ids, attention_mask, labels (lists)
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),  # pass custom optimizer and scheduler
)

# -------------------------
# Train
# -------------------------
print("Starting training...")
trainer.train()
print("Training finished.")

# -------------------------
# Save LoRA adapter + tokenizer
# -------------------------
print("Saving LoRA-adapted model to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR, save_embedding_layers=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved.")

