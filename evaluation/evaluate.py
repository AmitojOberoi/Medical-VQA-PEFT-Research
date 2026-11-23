import json
from pathlib import Path

from evaluation.metrics import compute_metrics


def evaluate_predictions(prediction_file):
    """
    Load prediction records from a JSON file
    and calculate evaluation metrics.
    """
    prediction_file = Path(prediction_file)

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    with prediction_file.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected a list of prediction records.")

    metrics = compute_metrics(records)

    return metrics


def print_results(dataset_name, metrics):
    """Display formatted evaluation results."""

    print(f"\n{'=' * 45}")
    print(f"       {dataset_name} EVALUATION")
    print(f"{'=' * 45}")

    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")

    print("=" * 45)
