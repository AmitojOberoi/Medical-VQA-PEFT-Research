
"""
preprocessing/validate_processed_data.py

Integrity validation for processed Medical VQA datasets.

Checks:
1. Preparation report availability and readability.
2. Expected annotation files.
3. JSONL readability and record structure.
4. Required fields and dataset/split labels.
5. Non-empty questions and answers.
6. Image path safety and existence.
7. Actual image decoding using Pillow.
8. Record counts against preparation_report.json.

Run:
    python -m preprocessing.validate_processed_data
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"
PREPARATION_REPORT = PROCESSED_DIR / "preparation_report.json"

VALIDATION_REPORT = PROCESSED_DIR / "validation_report.json"

DATASETS = {
    "SLAKE": "slake",
    "VQA_RAD": "vqa_rad",
    "PathVQA": "pathvqa",
}

SPLITS = ("train", "validation", "test")

REQUIRED_FIELDS = {
    "dataset",
    "split",
    "image_path",
    "question",
    "answer",
    "question_type",
    "answer_type",
    "modality",
    "source_id",
    "source_image_id",
}

MAX_DISPLAYED_ERRORS = 30


# --------------------------------------------------
# 2. ERROR HANDLING
# --------------------------------------------------

def add_error(errors, message):
    """Add an error to the validation report."""

    errors.append(message)


# --------------------------------------------------
# 3. SAFE IMAGE PATH RESOLUTION
# --------------------------------------------------

def resolve_image_path(image_path):
    """
    Resolve an annotation image path safely.

    Paths must remain inside the project directory.
    """

    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("Image path is missing or invalid.")

    relative_path = Path(image_path)

    if relative_path.is_absolute():
        raise ValueError("Absolute image paths are not allowed.")

    resolved_path = (PROJECT_ROOT / relative_path).resolve()

    try:
        resolved_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError("Image path escapes the project directory.")

    return resolved_path


# --------------------------------------------------
# 4. IMAGE DECODING
# --------------------------------------------------

def validate_image(image_path):
    """
    Verify that an image exists and can be decoded.

    Image.verify() checks file integrity without loading
    the complete image into memory.
    """

    if not image_path.is_file():
        return False, "Image file does not exist."

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True, None

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as error:

        return False, f"Image cannot be decoded: {error}"

    except Exception as error:

        return False, f"Unexpected image validation error: {error}"


# --------------------------------------------------
# 5. LOAD PREPARATION REPORT
# --------------------------------------------------

def load_preparation_report():
    """Load expected dataset record counts."""

    if not PREPARATION_REPORT.is_file():
        raise FileNotFoundError(
            f"Preparation report not found: {PREPARATION_REPORT}"
        )

    with PREPARATION_REPORT.open(
        "r", encoding="utf-8"
    ) as file:

        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("Preparation report must be a JSON object.")

    datasets = report.get("datasets")

    if not isinstance(datasets, dict):
        raise ValueError(
            "Preparation report does not contain a valid datasets object."
        )

    for dataset_name in DATASETS:

        if dataset_name not in datasets:
            raise ValueError(
                f"Missing dataset in preparation report: {dataset_name}"
            )

        for split in SPLITS:

            count_key = f"{split}_records"
            expected_count = datasets[dataset_name].get(count_key)

            if (
                not isinstance(expected_count, int)
                or isinstance(expected_count, bool)
                or expected_count < 0
            ):
                raise ValueError(
                    f"Invalid expected count: "
                    f"{dataset_name}.{count_key}"
                )

    return report


# --------------------------------------------------
# 6. VALIDATE ONE ANNOTATION FILE
# --------------------------------------------------

def validate_annotation_file(
    dataset_name,
    filename_prefix,
    split,
    expected_count,
    image_cache,
    decode_images=True,
):
    """Validate one processed JSONL annotation file."""

    annotation_file = (
        ANNOTATIONS_DIR / f"{filename_prefix}_{split}.jsonl"
    )

    result = {
        "dataset": dataset_name,
        "split": split,
        "file": str(
            annotation_file.relative_to(PROJECT_ROOT)
        ),
        "expected_records": expected_count,
        "actual_records": 0,
        "valid_records": 0,
        "images_checked": 0,
        "missing_images": 0,
        "invalid_images": 0,
        "errors": [],
        "status": "FAIL",
    }

    if not annotation_file.is_file():

        add_error(
            result["errors"],
            f"Annotation file missing: {annotation_file.name}",
        )

        return result

    seen_image_paths = set()

    try:
        with annotation_file.open(
            "r", encoding="utf-8-sig"
        ) as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    add_error(
                        result["errors"],
                        f"Line {line_number}: Empty JSONL line.",
                    )
                    continue

                result["actual_records"] += 1

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as error:

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Invalid JSON: {error}",
                    )
                    continue

                if not isinstance(record, dict):

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Record must be a JSON object.",
                    )
                    continue

                record_valid = True

                # Required field validation
                missing_fields = REQUIRED_FIELDS - set(record.keys())

                if missing_fields:

                    record_valid = False

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Missing fields: "
                        f"{sorted(missing_fields)}",
                    )

                # Dataset label validation
                if record.get("dataset") != dataset_name:

                    record_valid = False

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Incorrect dataset label.",
                    )

                # Split label validation
                if record.get("split") != split:

                    record_valid = False

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Incorrect split label.",
                    )

                # Question validation
                question = record.get("question")

                if not isinstance(question, str) or not question.strip():

                    record_valid = False

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Question is empty or invalid.",
                    )

                # Answer validation
                answer = record.get("answer")

                if not isinstance(answer, str) or not answer.strip():

                    record_valid = False

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Answer is empty or invalid.",
                    )

                # Image path validation
                try:

                    resolved_image = resolve_image_path(
                        record.get("image_path")
                    )

                    image_key = str(resolved_image)

                    if image_key not in seen_image_paths:

                        seen_image_paths.add(image_key)

                        if not resolved_image.is_file():

                            result["missing_images"] += 1
                            record_valid = False

                            image_cache[image_key] = (
                                False,
                                "Image file does not exist.",
                            )

                        elif decode_images:

                            if image_key not in image_cache:

                                image_cache[image_key] = validate_image(
                                    resolved_image
                                )

                            image_valid, image_error = image_cache[image_key]

                            result["images_checked"] += 1

                            if not image_valid:

                                result["invalid_images"] += 1
                                record_valid = False

                                add_error(
                                    result["errors"],
                                    f"Line {line_number}: "
                                    f"{image_error} "
                                    f"({record.get('image_path')})",
                                )

                    else:

                        image_valid, image_error = image_cache[image_key]

                        if not image_valid:
                            record_valid = False

                except ValueError as error:

                    record_valid = False

                    add_error(
                        result["errors"],
                        f"Line {line_number}: Invalid image path: {error}",
                    )

                if record_valid:
                    result["valid_records"] += 1

    except (OSError, UnicodeDecodeError) as error:

        add_error(
            result["errors"],
            f"Could not read annotation file: {error}",
        )

    # Compare actual and expected record counts
    if result["actual_records"] != expected_count:

        add_error(
            result["errors"],
            f"Record count mismatch: expected {expected_count}, "
            f"found {result['actual_records']}.",
        )

    if (
        result["actual_records"] == expected_count
        and result["valid_records"] == expected_count
        and not result["errors"]
    ):
        result["status"] = "PASS"

    return result


# --------------------------------------------------
# 7. VALIDATE ALL PROCESSED DATASETS
# --------------------------------------------------

def validate_all_processed_data(decode_images=True):
    """Run integrity checks across all datasets and splits."""

    print("\n" + "=" * 65)
    print("MEDICAL VQA PROCESSED DATA VALIDATION")
    print("=" * 65)

    preparation_report = load_preparation_report()

    image_cache = {}
    results = []

    for dataset_name, filename_prefix in DATASETS.items():

        print(f"\nDataset: {dataset_name}")
        print("-" * 50)

        for split in SPLITS:

            expected_count = preparation_report["datasets"][
                dataset_name
            ][f"{split}_records"]

            result = validate_annotation_file(
                dataset_name=dataset_name,
                filename_prefix=filename_prefix,
                split=split,
                expected_count=expected_count,
                image_cache=image_cache,
                decode_images=decode_images,
            )

            results.append(result)

            print(
                f"{split:12} "
                f"Expected: {result['expected_records']:6} | "
                f"Actual: {result['actual_records']:6} | "
                f"Valid: {result['valid_records']:6} | "
                f"{result['status']}"
            )

    # Overall summary
    total_expected = sum(
        result["expected_records"] for result in results
    )

    total_actual = sum(
        result["actual_records"] for result in results
    )

    total_valid = sum(
        result["valid_records"] for result in results
    )

    failed_splits = sum(
        result["status"] != "PASS" for result in results
    )

    all_errors = [
        error
        for result in results
        for error in result["errors"]
    ]

    overall_status = "PASS" if failed_splits == 0 else "FAIL"

    report = {
        "project": preparation_report.get("project"),
        "validation": {
            "status": overall_status,
            "image_decoding_enabled": decode_images,
            "datasets_checked": len(DATASETS),
            "splits_checked": len(results),
            "failed_splits": failed_splits,
            "expected_records": total_expected,
            "actual_records": total_actual,
            "valid_records": total_valid,
            "unique_images_checked": len(image_cache),
            "total_errors": len(all_errors),
        },
        "results": results,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with VALIDATION_REPORT.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Expected records:       {total_expected}")
    print(f"Actual records:         {total_actual}")
    print(f"Valid records:          {total_valid}")
    print(f"Unique images checked:  {len(image_cache)}")
    print(f"Failed splits:          {failed_splits}")
    print(f"Total errors:           {len(all_errors)}")

    print(f"\nOverall status: {overall_status}")
    print(f"Report saved: {VALIDATION_REPORT}")

    if all_errors:

        print("\nValidation errors:")

        for error in all_errors[:MAX_DISPLAYED_ERRORS]:
            print(f"  - {error}")

        remaining = len(all_errors) - MAX_DISPLAYED_ERRORS

        if remaining > 0:
            print(f"  ... and {remaining} more errors.")

    return report


# --------------------------------------------------
# 8. MAIN ENTRY POINT
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Validate processed Medical VQA datasets."
    )

    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Check image existence without decoding image contents.",
    )

    args = parser.parse_args()

    try:

        report = validate_all_processed_data(
            decode_images=not args.skip_image_decode
        )

        if report["validation"]["status"] != "PASS":
            sys.exit(1)

    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:

        print(f"\nVALIDATION COULD NOT COMPLETE: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
