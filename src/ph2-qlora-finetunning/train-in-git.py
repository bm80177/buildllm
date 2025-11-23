import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import random
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

# Login to HF hub
login(HF_TOKEN)

# -----------------------------
# Synthetic dataset
# -----------------------------
instructions = [
    "Customer asks about refund window",
    "Customer wants to cancel an order",
    "Order arrived late",
    "Wrong item received",
    "Product not working",
    "Shipping cost inquiry",
    "Change delivery address",
    "Request for invoice",
    "Ask about warranty",
    "Technical support request"
]

responses = [
    "Our refund window is 30 days from delivery.",
    "You can cancel your order from your account dashboard within 24 hours.",
    "Sorry for the delay. A delivery credit has been applied.",
    "We’ll ship the correct item and provide a return label.",
    "Please try resetting the product. Contact support if the issue persists.",
    "Shipping cost depends on your location and chosen delivery speed.",
    "You can update your delivery address before the order ships.",
    "An invoice will be emailed to you after purchase.",
    "Your product comes with a 12-month warranty.",
    "Our tech support team will contact you shortly."
]

train_data = []
for i in range(300):
    idx = random.randint(0, len(instructions)-1)
    train_data.append({"instruction": f"{instructions[idx]} #{i+1}", "response": responses[idx]})

dataset = Dataset.from_list(train_data)

# -----------------------------
# Model + Tokenizer
# -----------------------------
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"

# BitsAndBytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# Preprocessing
def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    enc = tokenizer(prompt, padding="max_length", truncation=True, max_length=256)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_dataset = dataset.map(preprocess)
tokenized_dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# LoRA config
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)

# Training
training_args = TrainingArguments(
    output_dir="./phi2-lora",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=True
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()

# Save adapter only (small)
model.save_pretrained("./phi2-lora")
tokenizer.save_pretrained("./phi2-lora")

# Push to Hugging Face Hub
model.push_to_hub("mishrabp/phi2-lora-finetuned", use_auth_token=HF_TOKEN)
tokenizer.push_to_hub("mishrabp/phi2-lora-finetuned", use_auth_token=HF_TOKEN)
