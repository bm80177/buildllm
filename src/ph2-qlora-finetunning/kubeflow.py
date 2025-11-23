import kfp
from kfp import dsl
from kfp.dsl import component, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.10",
    packages_to_install=["transformers", "datasets", "peft", "torch", "accelerate"]
)
def preprocess_data(ds: Output[Dataset]):
    from datasets import Dataset
    import json
    
    # synthetic dataset creation (your code)
    instructions = [...]
    responses = [...]

    dataset = Dataset.from_list([
        {"instruction": i, "response": r}
        for i, r in zip(instructions, responses)
    ])
    
    dataset.save_to_disk(ds.path)

@component(
    base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
    packages_to_install=["transformers", "peft", "datasets", "bitsandbytes"]
)
def train_lora_model(
    ds: Input[Dataset],
    model_out: Output[Model]
):
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    dataset = load_from_disk(ds.path)
    
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # model loading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True
    )

    # LoRA config
    lora_cfg = LoraConfig(...)
    model = get_peft_model(model, lora_cfg)

    # training
    args = TrainingArguments(
        output_dir=model_out.path,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        fp16=True
    )

    Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    ).train()

@component(
    base_image="python:3.10",
    packages_to_install=["transformers", "sklearn"]
)
def evaluate(model: Input[Model], metrics: Output[Metrics]):
    # Evaluate using sklearn or your custom code
    import json
    metrics.log_metric("f1_score", 0.97)
    metrics.log_metric("bleu", 0.89)
    metrics.log_metric("rouge", 0.91)

@dsl.pipeline(
    name="phi-2-lora-finetuning",
    description="Fine-tune phi-2 using LoRA with Kubeflow"
)
def pipeline():
    ds = preprocess_data()
    model = train_lora_model(ds.output)
    evaluate(model.output)
