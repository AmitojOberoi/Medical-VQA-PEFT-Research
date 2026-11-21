
import json
import pytest

from evaluate_pipeline import (
    compute_vqa_metrics,
    save_evaluation_results,
)


# 1. Test perfect predictions
def test_compute_vqa_metrics_perfect_predictions():
    predictions = ["MRI", "Yes", "No"]
    references = ["MRI", "Yes", "No"]

    metrics = compute_vqa_metrics(predictions, references)

    assert metrics["exact_match_accuracy"] == pytest.approx(1.0)
    assert metrics["correct_predictions"] == 3
    assert metrics["total_samples"] == 3


# 2. Test partially correct predictions
def test_compute_vqa_metrics_partial_accuracy():
    predictions = ["MRI", "No", "Yes", "CT"]
    references = ["MRI", "Yes", "Yes", "MRI"]

    metrics = compute_vqa_metrics(predictions, references)

    assert metrics["exact_match_accuracy"] == pytest.approx(0.5)
    assert metrics["correct_predictions"] == 2
    assert metrics["total_samples"] == 4


# 3. Test case-insensitive comparison
def test_compute_vqa_metrics_case_insensitive():
    predictions = ["mri", "YES"]
    references = ["MRI", "yes"]

    metrics = compute_vqa_metrics(predictions, references)

    assert metrics["exact_match_accuracy"] == pytest.approx(1.0)


# 4. Test whitespace handling
def test_compute_vqa_metrics_whitespace():
    predictions = [" MRI ", " Yes"]
    references = ["MRI", "Yes "]

    metrics = compute_vqa_metrics(predictions, references)

    assert metrics["exact_match_accuracy"] == pytest.approx(1.0)


# 5. Test empty inputs
def test_compute_vqa_metrics_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        compute_vqa_metrics([], [])


# 6. Test mismatched input lengths
def test_compute_vqa_metrics_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_vqa_metrics(["MRI"], ["MRI", "CT"])


# 7. Test invalid input types
def test_compute_vqa_metrics_invalid_types():
    with pytest.raises(ValueError, match="strings"):
        compute_vqa_metrics(["MRI", None], ["MRI", "CT"])


# 8. Test saving evaluation results
def test_save_evaluation_results(tmp_path):
    results = {
        "dataset": "SLAKE",
        "exact_match_accuracy": 0.85,
        "total_samples": 100,
    }

    output_path = tmp_path / "metrics" / "slake_results.json"

    saved_path = save_evaluation_results(results, output_path)

    assert saved_path == output_path
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded["dataset"] == "SLAKE"
    assert loaded["exact_match_accuracy"] == 0.85


# 9. Test invalid result object
def test_save_evaluation_results_invalid_input(tmp_path):
    with pytest.raises(ValueError, match="dictionary"):
        save_evaluation_results(
            ["invalid"],
            tmp_path / "results.json",
        )
