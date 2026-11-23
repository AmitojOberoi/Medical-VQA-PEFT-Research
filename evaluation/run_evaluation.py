from pathlib import Path
import json

from evaluation.evaluate import evaluate_predictions, print_results


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILES = {
    "PathVQA": PROJECT_ROOT / "results/experiment_1/predictions/pathvqa_500_results.json",
    "SLAKE": PROJECT_ROOT / "results/experiment_1/predictions/slake_results.json",
    "VQA-RAD": PROJECT_ROOT / "results/experiment_1/predictions/vqa_rad_results.json"
}

OUTPUT_DIR = PROJECT_ROOT / "results/experiment_1/metrics"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name, prediction_file in PREDICTION_FILES.items():
        print(f"\nEvaluating {dataset_name}...")

        metrics = evaluate_predictions(prediction_file)
        all_results[dataset_name] = metrics

        print_results(dataset_name, metrics)

    output_file = OUTPUT_DIR / "experiment_1_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print(f"\nCombined results saved to: {output_file}")


if __name__ == "__main__":
    main()
