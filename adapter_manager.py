import argparse
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "adapters.yaml"


def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Adapter registry not found: {REGISTRY_PATH}")

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file) or {}

    return registry.get("adapters", {})


def list_adapters():
    adapters = load_registry()

    if not adapters:
        print("No adapters found in registry.")
        return

    print("\nRegistered LoRA adapters:")
    for key, adapter in adapters.items():
        print(f"\n{key}")
        print(f"  Name: {adapter.get('name')}")
        print(f"  Path: {adapter.get('path')}")
        print(f"  Base model: {adapter.get('base_model')}")


def validate_adapter(adapter_name, adapter):
    adapter_path = PROJECT_ROOT / adapter["path"]
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"

    print(f"\nChecking {adapter.get('name', adapter_name)}...")

    if not adapter_path.is_dir():
        print("  FAIL: Adapter directory missing.")
        return False

    if not config_path.is_file():
        print("  FAIL: adapter_config.json missing.")
        return False

    if not weights_path.is_file():
        print("  FAIL: adapter_model.safetensors missing.")
        return False

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"  FAIL: Could not read adapter config: {error}")
        return False

    expected_model = adapter.get("base_model")
    actual_model = config.get("base_model_name_or_path")

    if expected_model != actual_model:
        print("  FAIL: Base model mismatch.")
        print(f"  Registry: {expected_model}")
        print(f"  Adapter:  {actual_model}")
        return False

    if config.get("peft_type") != "LORA":
        print(f"  FAIL: Expected LoRA, found {config.get('peft_type')}.")
        return False

    print("  PASS: Adapter files and configuration verified.")
    print(f"  LoRA rank: {config.get('r')}")
    print(f"  Alpha: {config.get('lora_alpha')}")
    print(f"  Target modules: {config.get('target_modules')}")

    return True


def validate_all():
    adapters = load_registry()
    results = []

    for name, adapter in adapters.items():
        results.append(validate_adapter(name, adapter))

    passed = sum(results)
    total = len(results)

    print(f"\nValidation summary: {passed}/{total} adapters passed.")

    if passed != total:
        raise SystemExit(1)

    print("All registered adapters are valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical VQA adapter manager")
    parser.add_argument("--list", action="store_true", help="List registered adapters")
    parser.add_argument("--validate", action="store_true", help="Validate all adapters")
    args = parser.parse_args()

    if args.list:
        list_adapters()
    elif args.validate:
        validate_all()
    else:
        parser.print_help()
