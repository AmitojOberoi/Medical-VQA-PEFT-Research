import re
from collections import Counter


def normalize_answer(text):
    """Convert an answer to lowercase and remove punctuation."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def calculate_exact_match(prediction, ground_truth):
    """Calculate normalized Exact Match."""
    return float(
        normalize_answer(prediction) == normalize_answer(ground_truth)
    )


def calculate_token_f1(prediction, ground_truth):
    """Calculate token-level F1 using token overlap."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common_tokens.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_metrics(records):
    """Calculate evaluation metrics for prediction records."""
    if not records:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    exact_matches = []
    f1_scores = []
    empty_predictions = 0

    for record in records:
        prediction = record["prediction"]
        ground_truth = record["ground_truth"]

        if not str(prediction).strip():
            empty_predictions += 1

        exact_matches.append(
            calculate_exact_match(prediction, ground_truth)
        )

        f1_scores.append(
            calculate_token_f1(prediction, ground_truth)
        )

    total = len(records)

    exact_match = sum(exact_matches) / total * 100
    token_f1 = sum(f1_scores) / total * 100

    return {
        "total_samples": total,
        "exact_match": round(exact_match, 2),
        "normalized_exact_match": round(exact_match, 2),
        "token_f1": round(token_f1, 2),
        "empty_predictions": empty_predictions
    }
