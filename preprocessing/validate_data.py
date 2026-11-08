"""
preprocessing/validate_data.py

Initial dataset inventory and integrity checks for the
Medical VQA PEFT research project.
"""

import argparse
import csv
import json
from pathlib import Path

from utils import PROJECT_ROOT, load_config


# --------------------------------------------------
# 1. DATASET FILE TYPES
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

METADATA_EXTENSIONS = {
    ".csv", ".json", ".jsonl"
}


# --------------------------------------------------
# 2. COUNT IMAGE FILES
# --------------------------------------------------

def count_images(dataset_path):
    """Count image files recursively inside a dataset folder."""

    return sum(
        1
        for file_path in dataset_path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------
# 3. INSPECT CSV FILES
# --------------------------------------------------

def inspect_csv(file_path):
    """Inspect CSV readability, columns, missing cells and duplicates."""

    with file_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        row_count = 0
        missing_cells = 0
        seen_rows = set()
        duplicate_rows = 0

        for row in reader:
            row_count += 1

            if any(
                value is None or not value.strip()
                for value in row.values()
            ):
                missing_cells += 1

            row_signature = tuple(
                (key, value) for key, value in row.items()
            )

            if row_signature in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(row_signature)

    return {
        "format": "CSV",
        "readable": True,
        "rows": row_count,
        "columns": columns,
        "missing_rows": missing_cells,
        "duplicate_rows": duplicate_rows
    }


# --------------------------------------------------
# 4. INSPECT JSON FILES
# --------------------------------------------------

def count_json_records(data):
    """
    Count records in common JSON structures.

    Supports a list of records or dictionaries containing
    lists of records. Does not assume a particular dataset schema.
    """

    if isinstance(data, list):
        return len(data)

    if isinstance(data, dict):
        list_lengths = [
            len(value)
            for value in data.values()
            if isinstance(value, list)
        ]

        return sum(list_lengths) if list_lengths else 1

    return 0


def inspect_json(file_path):
    """Inspect JSON readability and basic structure."""

    with file_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, dict):
        top_level_type = "object"
        top_level_keys = list(data.keys())

    elif isinstance(data, list):
        top_level_type = "array"
        top_level_keys = []

    else:
        top_level_type = type(data).__name__
        top_level_keys = []

    return {
        "format": "JSON",
        "readable": True,
        "top_level_type": top_level_type,
        "top_level_keys": top_level_keys,
        "records": count_json_records(data)
    }


# --------------------------------------------------
# 5. INSPECT JSONL FILES
# --------------------------------------------------

def inspect_jsonl(file_path):
    """Validate each non-empty JSON Lines record."""

    total_lines = 0
    valid_lines = 0
    invalid_lines = []

    with file_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            total_lines += 1

            try:
                json.loads(line)
                valid_lines += 1

            except json.JSONDecodeError:
                invalid_lines.append(line_number)

    return {
        "format": "JSONL",
        "readable": len(invalid_lines) == 0,
        "total_records": total_lines,
        "valid_records": valid_lines,
        "invalid_line_numbers": invalid_lines
    }


# --------------------------------------------------
# 6. INSPECT METADATA FILES
# --------------------------------------------------

def inspect_metadata(file_path):
    """Choose the correct inspector based on file extension."""

    extension = file_path.suffix.lower()

    if extension == ".csv":
        return inspect_csv(file_path)

    if extension == ".json":
        return inspect_json(file_path)

    if extension == ".jsonl":
        return inspect_jsonl(file_path)

    return {
        "format": extension,
        "readable": False,
        "error": "Unsupported metadata format"
    }


# --------------------------------------------------
# 7. VALIDATE ONE DATASET
# --------------------------------------------------

def validate_dataset(dataset_name, dataset_path):
    """Generate an inventory and validation summary."""

    print(f"\nInspecting dataset: {dataset_name}")

    result = {
        "dataset": dataset_name,
        "path": str(dataset_path),
        "status": None,
        "total_files": 0,
        "image_count": 0,
        "metadata_files": [],
        "errors": []
    }

    if not dataset_path.exists():
        result["status"] = "NOT_READY"
        result["errors"].append("Dataset directory does not exist.")

        print("  Dataset directory not found.")
        return result

    all_files = [
        path for path in dataset_path.rglob("*")
        if path.is_file()
    ]

    result["total_files"] = len(all_files)
    result["image_count"] = count_images(dataset_path)

    metadata_files = [
        path for path in all_files
        if path.suffix.lower() in METADATA_EXTENSIONS
    ]

    for metadata_path in metadata_files:

        try:
            details = inspect_metadata(metadata_path)

            details["file"] = str(
                metadata_path.relative_to(PROJECT_ROOT)
            )

            result["metadata_files"].append(details)

            if not details["readable"]:
                result["errors"].append(
                    f"Invalid metadata: {metadata_path.name}"
                )

        except (OSError, UnicodeDecodeError, csv.Error,
                json.JSONDecodeError) as error:

            result["errors"].append(
                f"{metadata_path.name}: {str(error)}"
            )

    if result["total_files"] == 0:
        result["status"] = "EMPTY"

    elif result["errors"]:
        result["status"] = "CHECK_REQUIRED"

    elif not metadata_files:
        result["status"] = "PARTIAL"

    else:
        result["status"] = "INVENTORIED"

    print(f"  Files: {result['total_files']}")
    print(f"  Images: {result['image_count']}")
    print(f"  Metadata files: {len(metadata_files)}")
    print(f"  Status: {result['status']}")

    return result


# --------------------------------------------------
# 8. VALIDATE ALL CONFIGURED DATASETS
# --------------------------------------------------

def validate_all_datasets(config):
    """Validate all enabled datasets and save a JSON report."""

    results = []

    for dataset_name, dataset_config in config["datasets"].items():

        if not dataset_config.get("enabled", True):
            print(f"\nSkipping disabled dataset: {dataset_name}")
            continue

        dataset_path = PROJECT_ROOT / dataset_config["path"]

        result = validate_dataset(
            dataset_name,
            dataset_path
        )

        results.append(result)

    report = {
        "project": config["project"]["name"],
        "datasets_checked": len(results),
        "results": results
    }

    report_path = PROJECT_ROOT / "results" / "metrics"
    report_path.mkdir(parents=True, exist_ok=True)

    output_file = report_path / "dataset_validation.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n----------------------------------")
    print("DATASET VALIDATION SUMMARY")
    print("----------------------------------")

    for result in results:
        print(
            f"{result['dataset']}: {result['status']}"
        )

    print(f"\nReport saved to: {output_file}")

    return report


# --------------------------------------------------
# 9. MAIN ENTRY POINT
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Medical VQA datasets."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to project YAML configuration."
    )

    args = parser.parse_args()

    config = load_config(args.config)

    validate_all_datasets(config)


if __name__ == "__main__":
    main()