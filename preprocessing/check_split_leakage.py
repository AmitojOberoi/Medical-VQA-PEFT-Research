
import hashlib
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS = ROOT / "data" / "processed" / "annotations"
REPORT_DIR = ROOT / "data" / "processed"

OUTPUT_FILE = REPORT_DIR / "split_leakage_report.json"

SPLITS = ("train", "validation", "test")
MAX_EXAMPLES = 20


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_records(file_path):

    records = []

    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, "
                    f"line {line_number}: {error}"
                )

    return records


# ============================================================
# IMAGE HASHING
# ============================================================

def calculate_image_hash(image_path):

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    digest = hashlib.sha256()

    with image_path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(dataset_name):

    print("\n" + "=" * 65)
    print(f"ANALYZING IMAGE SPLITS: {dataset_name}")
    print("=" * 65)

    split_records = {}

    for split in SPLITS:

        file_path = ANNOTATIONS / f"{dataset_name.lower()}_{split}.jsonl"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing annotation file: {file_path}"
            )

        split_records[split] = load_records(file_path)

    # Track image paths and image content hashes.
    path_splits = defaultdict(set)
    hash_splits = defaultdict(set)

    path_records = defaultdict(int)
    hash_records = defaultdict(int)

    hash_examples = {}

    for split, records in split_records.items():

        for record in records:

            relative_path = record.get("image_path")

            if not relative_path:
                raise ValueError(
                    f"Missing image_path in {dataset_name} {split}"
                )

            image_path = (ROOT / relative_path).resolve()

            # Ensure the path stays inside the project directory.
            if not image_path.is_relative_to(ROOT.resolve()):
                raise ValueError(
                    f"Image path outside project: {relative_path}"
                )

            image_hash = calculate_image_hash(image_path)

            path_splits[relative_path].add(split)
            hash_splits[image_hash].add(split)

            path_records[relative_path] += 1
            hash_records[image_hash] += 1

            hash_examples[image_hash] = relative_path

    # Find image paths appearing in multiple splits.
    path_leakage = {
        path: sorted(splits)
        for path, splits in path_splits.items()
        if len(splits) > 1
    }

    # Find identical image contents appearing in multiple splits,
    # even if they have different file paths.
    hash_leakage = {
        image_hash: {
            "splits": sorted(splits),
            "example_path": hash_examples[image_hash],
        }
        for image_hash, splits in hash_splits.items()
        if len(splits) > 1
    }

    result = {
        "dataset": dataset_name,

        "split_record_counts": {
            split: len(records)
            for split, records in split_records.items()
        },

        "unique_image_paths": len(path_splits),

        "unique_image_contents": len(hash_splits),

        "path_overlap_groups": len(path_leakage),

        "content_overlap_groups": len(hash_leakage),

        "path_overlap_examples": [
            {
                "image_path": path,
                "splits": splits,
            }
            for path, splits in list(
                path_leakage.items()
            )[:MAX_EXAMPLES]
        ],

        "content_overlap_examples": [
            {
                "image_hash": image_hash,
                **details,
            }
            for image_hash, details in list(
                hash_leakage.items()
            )[:MAX_EXAMPLES]
        ],

        "status": (
            "PASS"
            if not path_leakage and not hash_leakage
            else "FAIL"
        ),
    }

    print(
        "Unique image paths:",
        result["unique_image_paths"]
    )

    print(
        "Unique image contents:",
        result["unique_image_contents"]
    )

    print(
        "Images shared across splits:",
        result["path_overlap_groups"]
    )

    print(
        "Identical image contents shared across splits:",
        result["content_overlap_groups"]
    )

    print("Status:", result["status"])

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("MEDICAL VQA IMAGE-LEVEL SPLIT LEAKAGE CHECK")
    print("=" * 65)

    datasets = ["SLAKE", "VQA_RAD", "PathVQA"]

    report = {
        "project": "Medical-VQA-PEFT-Research",
        "method": (
            "Cross-split comparison using image paths "
            "and SHA256 image-content hashes."
        ),
        "datasets": {},
    }

    overall_pass = True

    for dataset in datasets:

        result = analyze_dataset(dataset)

        report["datasets"][dataset] = result

        if result["status"] != "PASS":
            overall_pass = False

    report["overall_status"] = (
        "PASS" if overall_pass else "REVIEW REQUIRED"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("FINAL LEAKAGE CHECK")
    print("=" * 65)

    print("Overall status:", report["overall_status"])
    print("Report saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
