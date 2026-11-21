
"""
peft_methods.py

LoRA configuration and utilities for Medical VQA.

Final experimental method: LoRA only.
"""

from peft import LoraConfig, get_peft_model


def create_lora_config(config):
    """Create LoRA configuration from project YAML settings."""

    peft_config = config["peft"]

    if peft_config.get("method") != "lora":
        raise ValueError("This project uses LoRA only.")

    lora = peft_config.get("lora", {})

    if not lora.get("enabled", True):
        raise ValueError("LoRA is disabled in the configuration.")

    return LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get(
            "target_modules", ["q_proj", "v_proj"]
        ),
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
    )


def apply_lora(model, config):
    """Apply LoRA to a base model."""

    lora_config = create_lora_config(config)

    model = get_peft_model(model, lora_config)

    # Keep trainable adapter parameters in FP32.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name:
            parameter.data = parameter.data.float()

    model.print_trainable_parameters()

    return model


def get_trainable_parameter_summary(model):
    """Return total and trainable parameter counts."""

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
            (trainable / total) * 100 if total else 0
        ),
    }
