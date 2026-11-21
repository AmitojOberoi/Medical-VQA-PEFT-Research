
"""
Evaluation utilities for Medical VQA.
"""

import json
from pathlib import Path


def compute_vqa_metrics(predictions, references):
    """
    Calculate exact-match accuracy for VQA predictions.

    Args:
        predictions: List of predicted answers.
        references: List of ground-truth answers.

    Returns:
        Dictionary containing evaluation metrics.
    """

    if not predictions or not references:
        raise ValueError("Predictions and references cannot be empty.")

    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have the same length."
        )

    if not all(isinstance(x, str) for x in predictions + references):
        raise ValueError("Predictions and references must contain strings.")

    normalized_predictions = [
        answer.strip().casefold() for answer in predictions
    ]

    normalized_references = [
        answer.strip().casefold() for answer in references
    ]

    correct = sum(
        prediction == reference
        for prediction, reference in zip(
            normalized_predictions,
            normalized_references,
        )
    )

    total = len(references)

    return {
        "exact_match_accuracy": correct / total,
        "correct_predictions": correct,
        "total_samples": total,
    }


def save_evaluation_results(results, output_path):
    """
    Save evaluation metrics as a JSON file.

    Args:
        results: Dictionary containing evaluation results.
        output_path: Destination JSON file path.

    Returns:
        Path to the saved file.
    """

    if not isinstance(results, dict):
        raise ValueError("Evaluation results must be a dictionary.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    return output_path
