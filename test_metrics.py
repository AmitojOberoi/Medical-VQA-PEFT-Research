from evaluation.metrics import (
    calculate_exact_match,
    calculate_token_f1,
    compute_metrics
)

assert calculate_exact_match("YES", "yes") == 1.0
assert calculate_exact_match("histones", "histone subunits") == 0.0

assert calculate_token_f1("yes", "yes") == 1.0
assert calculate_token_f1(
    "the histones", "the histone subunits"
) > 0.0

sample_records = [
    {"ground_truth": "yes", "prediction": "yes"},
    {"ground_truth": "histone subunits", "prediction": "histones"}
]

metrics = compute_metrics(sample_records)

print("Metric tests passed.")
print(metrics)
