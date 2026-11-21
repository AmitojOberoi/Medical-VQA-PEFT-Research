
import pytest
import torch
from torch import nn

pytest.importorskip("peft")

import peft_methods
from peft_methods import (
    create_lora_config,
    apply_lora,
    get_trainable_parameter_summary,
)


@pytest.fixture
def valid_config():
    return {
        "peft": {
            "method": "lora",
            "lora": {
                "enabled": True,
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        }
    }


def test_create_lora_config(valid_config):
    config = create_lora_config(valid_config)

    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) == {"q_proj", "v_proj"}


def test_lora_method_is_required(valid_config):
    valid_config["peft"]["method"] = "qlora"

    with pytest.raises(ValueError, match="LoRA only"):
        create_lora_config(valid_config)


def test_lora_must_be_enabled(valid_config):
    valid_config["peft"]["lora"]["enabled"] = False

    with pytest.raises(ValueError, match="LoRA is disabled"):
        create_lora_config(valid_config)


def test_lora_configuration_uses_expected_defaults():
    config = create_lora_config({
        "peft": {
            "method": "lora",
            "lora": {},
        }
    })

    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05


def test_parameter_summary():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Linear(5, 2),
    )

    summary = get_trainable_parameter_summary(model)

    expected_total = sum(p.numel() for p in model.parameters())

    assert summary["total_parameters"] == expected_total
    assert summary["trainable_parameters"] == expected_total
    assert summary["trainable_percentage"] == pytest.approx(100.0)


def test_parameter_summary_with_frozen_parameters():
    model = nn.Linear(10, 5)
    model.weight.requires_grad = False

    summary = get_trainable_parameter_summary(model)

    assert summary["total_parameters"] == 55
    assert summary["trainable_parameters"] == 5
    assert summary["trainable_percentage"] == pytest.approx(
        5 / 55 * 100
    )


def test_apply_lora_uses_config(valid_config, monkeypatch):
    model = nn.Linear(4, 4)
    called = {}

    def fake_get_peft_model(base_model, lora_config):
        called["model"] = base_model
        called["config"] = lora_config
        return base_model

    monkeypatch.setattr(
        peft_methods,
        "get_peft_model",
        fake_get_peft_model,
    )

    monkeypatch.setattr(
        model,
        "print_trainable_parameters",
        lambda: None,
        raising=False,
    )

    result = apply_lora(model, valid_config)

    assert result is model
    assert called["model"] is model
    assert called["config"].r == 8


def test_trainable_parameter_summary_has_expected_keys():
    model = nn.Linear(3, 2)

    summary = get_trainable_parameter_summary(model)

    assert set(summary.keys()) == {
        "total_parameters",
        "trainable_parameters",
        "trainable_percentage",
    }


def test_parameter_summary_handles_zero_parameters():
    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()

    summary = get_trainable_parameter_summary(EmptyModel())

    assert summary["total_parameters"] == 0
    assert summary["trainable_parameters"] == 0
    assert summary["trainable_percentage"] == 0
