
"""
utils.py

Shared utility functions for the Medical VQA PEFT research project.
"""

import random
from pathlib import Path

import yaml


# --------------------------------------------------
# 1. PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. LOAD CONFIGURATION
# --------------------------------------------------

def load_config(config_path="configs/config.yaml"):
    """
    Load project configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Project configuration.
    """

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a valid YAML mapping."
        )

    required_sections = [
        "project",
        "paths",
        "datasets",
        "experiments"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}"
            )

    return config


# --------------------------------------------------
# 3. CREATE PROJECT DIRECTORIES
# --------------------------------------------------

def create_project_dirs(config):
    """
    Create project directories defined in config.yaml.

    Args:
        config (dict): Project configuration.
    """

    for relative_path in config["paths"].values():
        directory = PROJECT_ROOT / relative_path
        directory.mkdir(parents=True, exist_ok=True)

    for dataset in config["datasets"].values():
        directory = PROJECT_ROOT / dataset["path"]
        directory.mkdir(parents=True, exist_ok=True)

    print("Project directories verified successfully.")


# --------------------------------------------------
# 4. SET RANDOM SEED
# --------------------------------------------------

def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.

    Supports Python, NumPy and PyTorch when installed.
    """

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)

    except ImportError:
        print("NumPy not installed. Skipping NumPy seed.")

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        print("PyTorch not installed. Skipping PyTorch seed.")

    print(f"Random seed set to {seed}.")


# --------------------------------------------------
# 5. DETECT COMPUTING DEVICE
# --------------------------------------------------

def get_device():
    """
    Detect CUDA GPU availability.

    Returns:
        str: 'cuda' if available, otherwise 'cpu'.
    """

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"

            print(
                f"GPU detected: {torch.cuda.get_device_name(0)}"
            )

        else:
            device = "cpu"
            print("No CUDA GPU detected. Using CPU.")

    except ImportError:
        device = "cpu"
        print("PyTorch not installed. Using CPU for now.")

    return device
