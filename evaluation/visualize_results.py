import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = ROOT / "results/experiment_1/metrics/experiment_1_results.json"
OUTPUT_DIR = ROOT / "results/experiment_1/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        results = json.load(f)

    datasets = list(results.keys())
    em = [results[d]["exact_match"] for d in datasets]
    f1 = [results[d]["token_f1"] for d in datasets]

    # Exact Match chart
    plt.figure(figsize=(9, 6))
    bars = plt.bar(datasets, em, width=0.55)
    plt.title("Exact Match Comparison Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Exact Match (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.bar_label(bars, fmt="%.2f%%", padding=4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exact_match_comparison.png", dpi=300)
    plt.close()

    # Token F1 chart
    plt.figure(figsize=(9, 6))
    bars = plt.bar(datasets, f1, width=0.55)
    plt.title("Token F1 Comparison Across Datasets")
    plt.xlabel("Dataset")
    plt.ylabel("Token F1 (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.bar_label(bars, fmt="%.2f%%", padding=4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "token_f1_comparison.png", dpi=300)
    plt.close()

    # Combined chart
    x = np.arange(len(datasets))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width / 2, em, width, label="Exact Match")
    bars2 = plt.bar(x + width / 2, f1, width, label="Token F1")

    plt.title("Experiment 1: Comparative Performance")
    plt.xlabel("Dataset")
    plt.ylabel("Score (%)")
    plt.xticks(x, datasets)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.bar_label(bars1, fmt="%.2f", padding=3)
    plt.bar_label(bars2, fmt="%.2f", padding=3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combined_comparison.png", dpi=300)
    plt.close()

    print("\nVisualizations generated successfully!")

    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"{file.name} ({file.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
