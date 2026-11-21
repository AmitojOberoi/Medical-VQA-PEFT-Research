
"""
model_loader.py

Loads the BLIP-2 vision-language model and processor.

Supports:
    - Standard FP16 loading
    - 4-bit loading for QLoRA experiments

Actual model loading is intended for Google Colab.
"""

from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_model_config(config_path=CONFIG_PATH):
    """Load and validate the project configuration."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    required_sections = ["project", "model", "training", "peft"]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    model_config = config["model"]

    if not model_config.get("model_id"):
        raise ValueError("model.model_id is not configured.")

    if model_config.get("model_family") != "blip2":
        raise ValueError("This loader currently supports BLIP-2 only.")

    return config


def get_torch_dtype(dtype_name):
    """Convert a configuration string into a PyTorch dtype."""

    supported_dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if dtype_name not in supported_dtypes:
        raise ValueError(
            f"Unsupported torch dtype: {dtype_name}"
        )

    return supported_dtypes[dtype_name]


def load_blip2_model(config=None, quantized=False):
    """
    Load BLIP-2 and its image-text processor.

    Args:
        config: Project configuration dictionary.
        quantized: Load the language model in 4-bit mode.

    Returns:
        model, processor
    """

    if config is None:
        config = load_model_config()

    model_config = config["model"]

    model_id = model_config["model_id"]

    # Quantized training requires a supported CUDA environment.
    if quantized and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit QLoRA loading requires a supported CUDA GPU. "
            "Run this experiment in Google Colab."
        )

    try:
        from transformers import (
            AutoProcessor,
            Blip2ForConditionalGeneration,
        )

    except ImportError as error:
        raise ImportError(
            "Transformers is required. "
            "Install it using: pip install transformers"
        ) from error

    dtype = get_torch_dtype(
        model_config.get("torch_dtype", "float16")
    )

    processor = AutoProcessor.from_pretrained(model_id)

    loading_arguments = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        loading_arguments["device_map"] = "auto"

    # Configure 4-bit quantization for QLoRA.
    if quantized:
        try:
            from transformers import BitsAndBytesConfig

        except ImportError as error:
            raise ImportError(
                "BitsAndBytes support is unavailable. "
                "Install a compatible bitsandbytes build "
                "in Google Colab."
            ) from error

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        loading_arguments["quantization_config"] = (
            quantization_config
        )

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        **loading_arguments,
    )

    if model_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model, processor


def count_parameters(model):
    """Return total and trainable parameter counts."""

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_percentage = (
        100 * trainable_parameters / total_parameters
        if total_parameters else 0
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": trainable_percentage,
    }


if __name__ == "__main__":

    configuration = load_model_config()

    print("Model configuration validated.")
    print("Model:", configuration["model"]["model_id"])
    print("Architecture:", configuration["model"]["architecture"])
    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("Model weights have NOT been downloaded.")
