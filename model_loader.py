
"""
model_loader.py

Loads the LLaVA-Med base model and optional LoRA adapter.

Final project scope:
    - LLaVA-Med 7B
    - LoRA fine-tuning
    - FP16 inference where CUDA is available
"""

from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_model_config(config_path=CONFIG_PATH):
    """Load and validate the project configuration."""

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = [
        "project",
        "model",
        "training",
        "peft",
        "datasets",
        "paths",
    ]

    for section in required:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    model_config = config["model"]

    if model_config.get("model_family") != "llava":
        raise ValueError("Only LLaVA model family is supported.")

    if config["peft"].get("method") != "lora":
        raise ValueError("Final project supports LoRA only.")

    if not model_config.get("model_id"):
        raise ValueError("Missing model.model_id.")

    return config


def get_torch_dtype(dtype_name):
    """Convert a dtype string to a PyTorch dtype."""

    supported = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if dtype_name not in supported:
        raise ValueError(
            f"Unsupported torch dtype: {dtype_name}"
        )

    if dtype_name == "bfloat16" and not torch.cuda.is_available():
        raise RuntimeError(
            "BFloat16 configuration requires a compatible GPU."
        )

    return supported[dtype_name]


def get_device():
    """Detect the available computing device."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_llava_model(
    config=None,
    adapter_path=None,
    device_map=None,
):
    """
    Load the LLaVA-Med base model and optionally a LoRA adapter.

    Returns:
        tokenizer, model, image_processor, context_length
    """

    if config is None:
        config = load_model_config()

    model_config = config["model"]
    model_id = model_config["model_id"]

    if device_map is None:
        device_map = (
            model_config.get("device_map", "auto")
            if torch.cuda.is_available()
            else "cpu"
        )

    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

    except ImportError as error:
        raise ImportError(
            "Official LLaVA code is unavailable. "
            "Install the LLaVA repository and its dependencies "
            "before loading the model."
        ) from error

    model_name = get_model_name_from_path(model_id)

    print("Loading LLaVA-Med base model...")
    print("Model:", model_id)
    print("Device:", device_map)

    tokenizer, model, image_processor, context_length = (
        load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name=model_name,
            device_map=device_map,
        )
    )

    if adapter_path is not None:
        adapter_path = Path(adapter_path)

        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {adapter_path}"
            )

        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"adapter_config.json missing in {adapter_path}"
            )

        try:
            from peft import PeftModel

        except ImportError as error:
            raise ImportError(
                "PEFT is required to load the LoRA adapter."
            ) from error

        print("Loading LoRA adapter:", adapter_path)

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    model.eval()

    return tokenizer, model, image_processor, context_length


def count_parameters(model):
    """Count total and trainable model parameters."""

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": (
            100 * trainable / total if total else 0
        ),
    }


if __name__ == "__main__":

    configuration = load_model_config()

    print("Model configuration validated.")
    print("Model:", configuration["model"]["model_id"])
    print("Architecture:", configuration["model"]["architecture"])
    print("PEFT method:", configuration["peft"]["method"])
    print("Device:", get_device())
    print("Model weights have NOT been downloaded.")
