
"""
colab_train.py

Multimodal PEFT training for Medical VQA using LLaVA-Med v1.5.

Supported methods:
    - lora
    - qlora

Run inside Google Colab with a CUDA GPU.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)



from llava.constants import IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model import LlavaMistralForCausalLM
MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"

DATASET_FILES = {
    "SLAKE": "slake",
    "VQA_RAD": "vqa_rad",
    "PathVQA": "pathvqa",
}

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "llava_format"
RESULTS_DIR = PROJECT_ROOT / "results" / "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(datasets, split, limit=None):
    """Load records from one or more converted dataset files."""

    records = []

    for dataset_name in datasets:
        prefix = DATASET_FILES[dataset_name]

        file_path = DATA_DIR / f"{prefix}_{split}.json"

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            dataset_records = json.load(file)

        for record in dataset_records:
            record["dataset_name"] = dataset_name

        records.extend(dataset_records)

    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError(
            f"No records loaded for split: {split}"
        )

    print(f"Loaded {len(records)} records for {split}.")

    return records


class MedicalVQATrainingDataset(Dataset):
    """Convert Medical VQA records into LLaVA model inputs."""

    def __init__(
        self,
        records,
        tokenizer,
        image_processor,
        model_config,
        max_length=1024,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.max_length = max_length

        self.conversation_template = conv_templates[
            "mistral_instruct"
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = (
            PROJECT_ROOT / record["image"]
        ).resolve()

        try:
            image_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"Image path escapes project: {image_path}"
            ) from error

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            original_size = image.size

            image_tensor = process_images(
                [image],
                self.image_processor,
                self.model_config,
            )[0]

        question = record["conversations"][0]["value"]
        answer = record["conversations"][1]["value"]

        # Remove the image marker from the question because
        # the conversation template adds it explicitly.
        question = question.replace("<image>", "").strip()

        # Build the full conversation.
        full_conversation = self.conversation_template.copy()

        full_conversation.append_message(
            full_conversation.roles[0],
            "<image>\n" + question,
        )

        full_conversation.append_message(
            full_conversation.roles[1],
            answer,
        )

        full_prompt = full_conversation.get_prompt()

        # Build the prompt-only version to mask question tokens.
        prompt_conversation = self.conversation_template.copy()

        prompt_conversation.append_message(
            prompt_conversation.roles[0],
            "<image>\n" + question,
        )

        prompt_conversation.append_message(
            prompt_conversation.roles[1],
            None,
        )

        prompt_only = prompt_conversation.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            return_tensors="pt",
        )

        prompt_ids = tokenizer_image_token(
            prompt_only,
            self.tokenizer,
            return_tensors="pt",
        )

        input_ids = input_ids[:self.max_length]
        prompt_length = min(
            prompt_ids.shape[0],
            input_ids.shape[0],
        )

        labels = input_ids.clone()

        # Ignore the question and image-token portion of the loss.
        labels[:prompt_length] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": image_tensor,
            "image_size": original_size,
        }


class MedicalVQACollator:
    """Pad text sequences and combine image tensors."""

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch):
        max_length = max(
            item["input_ids"].shape[0]
            for item in batch
        )

        batch_size = len(batch)

        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )

        labels = torch.full(
            (batch_size, max_length),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )

        for index, item in enumerate(batch):
            length = item["input_ids"].shape[0]

            input_ids[index, :length] = item["input_ids"]
            labels[index, :length] = item["labels"]
            attention_mask[index, :length] = 1

        images = torch.stack(
            [item["image"] for item in batch]
        )

        image_sizes = [
            item["image_size"]
            for item in batch
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
            "image_sizes": image_sizes,
        }


def load_model(method):
    """Load the official LLaVA-Med Mistral model."""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None

    if method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = LlavaMistralForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    else:
        model = LlavaMistralForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    # Load the official vision tower and image processor.
    vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()

    image_processor = vision_tower.image_processor

    model.config.use_cache = False

    if method == "lora":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    return model, tokenizer, image_processor


def apply_lora(model, method, config):
    """Apply LoRA adapters to the language model."""

    lora_config = config["peft"]["lora"]

    if method == "qlora":
        lora_config = config["peft"]["qlora"]

    peft_config = LoraConfig(
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.print_trainable_parameters()

    return model


def load_project_config():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Project configuration not found: {config_path}"
        )

    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["lora", "qlora"],
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=["all", "SLAKE", "VQA_RAD", "PathVQA"],
        default="all",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Run this script in "
            "Google Colab with a GPU runtime."
        )

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    config = load_project_config()

    set_seed(config["project"].get("seed", 42))

    datasets = (
        list(DATASET_FILES)
        if args.dataset == "all"
        else [args.dataset]
    )

    print("\nMedical VQA PEFT Training")
    print("------------------------")
    print(f"Model: {MODEL_ID}")
    print(f"Method: {args.method}")
    print(f"Datasets: {datasets}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer, image_processor = load_model(
        args.method
    )

    model = apply_lora(
        model,
        args.method,
        config,
    )

    train_records = load_records(
        datasets,
        "train",
        args.max_train_samples,
    )

    validation_records = load_records(
        datasets,
        "validation",
        args.max_val_samples,
    )

    train_dataset = MedicalVQATrainingDataset(
        train_records,
        tokenizer,
        image_processor,
        model.config,
        args.max_length,
    )

    validation_dataset = MedicalVQATrainingDataset(
        validation_records,
        tokenizer,
        image_processor,
        model.config,
        args.max_length,
    )

    collator = MedicalVQACollator(tokenizer)

    experiment_name = (
        f"{args.method}_{args.dataset.lower()}"
    )

    output_dir = RESULTS_DIR / experiment_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_config = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=float(
            training_config["learning_rate"]
        ),

        weight_decay=float(
            training_config.get("weight_decay", 0.01)
        ),

        warmup_ratio=float(
            training_config.get("warmup_ratio", 0.03)
        ),

        logging_steps=10,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,
        bf16=False,

        gradient_checkpointing=True,

        remove_unused_columns=False,

        report_to="none",

        dataloader_num_workers=2,

        optim=(
            "paged_adamw_8bit"
            if args.method == "qlora"
            else "adamw_torch"
        ),

        seed=config["project"].get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    print("\nStarting training...\n")

    training_result = trainer.train()

    training_time = time.time() - start_time

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    final_metrics = trainer.evaluate()

    results = {
        "model": MODEL_ID,
        "method": args.method,
        "datasets": datasets,
        "epochs": args.epochs,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "training_time_seconds": training_time,
        "peak_gpu_memory_gb": peak_memory_gb,
        "train_loss": training_result.training_loss,
        "validation_loss": final_metrics.get("eval_loss"),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    }

    results_path = output_dir / "training_results.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nTraining completed.")
    print("-------------------")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()

