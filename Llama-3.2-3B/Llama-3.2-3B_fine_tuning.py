# train_llama3_3b_windows.py
"""
Single-GPU Windows-compatible training script for Llama-3.2-3B with LoRA (PEFT).
Usage: python train_llama3_3b_windows.py
Requirements:
    pip install -U transformers datasets accelerate peft torch
Notes:
    - This script uses full precision (fp32) by default.
    - Modify MODEL_PATH/DATA_PATH/OUTPUT_DIR as needed.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"

import math
from typing import List, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# -------------------------
# User config (edit if needed)
# -------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B"
# If you have pre-downloaded model files, set MODEL_PATH to that folder.
# Otherwise set MODEL_PATH = None to download from HF hub.
MODEL_PATH = "D:/2MOE/Llama-3.2-3B/Llama-3.2-3B_model"
DATA_PATH = "D:/2MOE/fine_tuning_dataset/fine_tuning_dataset_oneshot.jsonl"
OUTPUT_DIR = "./llama3_3b_lora_output"
CUTOFF_LEN = 8192

# Hyperparameters (as requested)
LR = 5e-5
EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
MAX_GRAD_NORM = 1.0

# LoRA params
LORA_R = 256
LORA_ALPHA = 512
LORA_DROPOUT = 0.05
# typical target modules for Llama-style models; keep a broad set
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "down_proj", "up_proj", "gate_proj"
]

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)

# Optional: point to a mirror if you have slow HF access
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------
# Utility: build prompt from json example
# -------------------------
def build_prompt_from_example(ex: Dict) -> str:
    parts: List[str] = []
    passage = ex.get("passage")
    question = ex.get("question")
    options = ex.get("options")  # list or None

    if passage and str(passage).strip():
        parts.append("Context: " + str(passage).strip())
    if question and str(question).strip():
        parts.append("Question: " + str(question).strip())

    if options:
        parts.append("Options:")
        for o in options:
            parts.append(str(o).strip())

    # model should generate after "Answer:"
    parts.append("Answer:")

    return "\n".join(parts)

# -------------------------
# Load dataset
# -------------------------
print("Loading dataset:", DATA_PATH)
raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
print("Dataset loaded. Examples:", len(raw_ds))

# -------------------------
# Tokenizer & Model (full precision)
# -------------------------
print("Loading tokenizer and model (full precision fp32)...")
tokenizer_kwargs = {"trust_remote_code": True}  # keep for models that require it
if MODEL_PATH:
    tokenizer_kwargs["cache_dir"] = MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print("Added pad token to tokenizer.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_kwargs = {"torch_dtype": torch.float32, "trust_remote_code": True}
if MODEL_PATH:
    model_kwargs["cache_dir"] = MODEL_PATH

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir=MODEL_PATH)
# resize tokens if tokenizer expanded
model.resize_token_embeddings(len(tokenizer))
model.to(device)
print("Model loaded to device.")

# -------------------------
# Apply LoRA (PEFT)
# -------------------------
print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})")
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
def map_answer_to_label_text(example: Dict) -> str:
    """
    Given the example with 'answer' and 'options', try to return the option's text.
    If answer is missing, return empty string.
    """
    answer = example.get("answer")
    options = example.get("options")
    if not answer:
        return ""
    # answer might be list like ["A"] or ["(A) text"] or string "A"
    if isinstance(answer, list) and len(answer) > 0:
        ans = answer[0]
    else:
        ans = answer
    if ans is None:
        return ""
    ans = str(ans).strip()

    # if options exist, map letter to option text
    if isinstance(options, list) and len(options) > 0:
        # try matching by leading letter A/B/C etc
        for opt in options:
            opt_str = str(opt).strip()
            # common option starts: "(A) text", "A. text", "A) text", "A text"
            if opt_str.upper().startswith(ans.upper()):
                # remove leading label if present
                if opt_str.startswith("(") and ")" in opt_str:
                    return opt_str.split(")", 1)[1].strip()
                if "." in opt_str and opt_str.index(".") <= 3:
                    return opt_str.split(".", 1)[1].strip()
                if ")" in opt_str and opt_str.index(")") <= 3:
                    return opt_str.split(")", 1)[1].strip()
                # fallback: return whole option
                return opt_str
        # if not found, fallback to returning ans itself
        return ans
    else:
        return ans

def tokenize_example(example: Dict):
    prompt = build_prompt_from_example(example)
    label_text = map_answer_to_label_text(example)

    if label_text:
        text = prompt + " " + label_text
    else:
        text = prompt

    enc = tokenizer(
        text,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", [1] * len(input_ids))
    labels = input_ids.copy()

    if label_text:
        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
        )
        prompt_len = len(enc_prompt["input_ids"])
        # mark prompt tokens with -100
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
    else:
        # no label -> ignore loss
        labels = [-100] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

print("Tokenizing dataset (this may take time)...")
tokenized = raw_ds.map(tokenize_example, remove_columns=raw_ds.column_names)
print("Tokenization done. Tokenized examples:", len(tokenized))

# -------------------------
# Training step counts
# -------------------------
train_data_len = len(tokenized)
effective_batch = PER_DEVICE_BATCH_SIZE  # single GPU
steps_per_epoch = math.ceil(train_data_len / (effective_batch * GRADIENT_ACCUMULATION_STEPS))
total_training_steps = max(1, steps_per_epoch * EPOCHS)
print(f"Train size: {train_data_len}, steps_per_epoch: {steps_per_epoch}, total_steps: {total_training_steps}")

# -------------------------
# Optimizer & Scheduler
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
scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_training_steps))

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
    fp16=False,            # full precision
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
    optimizers=(optimizer, scheduler),
    # pass label_names to avoid warning about PeftModel hiding base model signature
    # label_names=["labels"],
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving LoRA-adapted model to", OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved.")
