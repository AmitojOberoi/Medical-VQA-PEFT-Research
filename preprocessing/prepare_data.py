
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

IMAGES = PROCESSED / "images"
ANNOTATIONS = PROCESSED / "annotations"

SEED = 42
PARQUET_BATCH_SIZE = 128


# ============================================================
# GENERAL UTILITIES
# ============================================================

def check_required_path(path):
    """Raise an error if a required file or directory is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset path not found: {path}"
        )


def prepare_output_directories():
    """Create output directories and clear old annotations."""

    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    # Remove only previously generated annotation files.
    for file in ANNOTATIONS.glob("*.jsonl"):
        file.unlink()


def write_record(record, output_file):
    """Write a standardized record to an open JSONL file."""

    output_file.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )


def copy_image(source, destination):
    """Copy an image into processed storage."""

    if not source.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        shutil.copy2(source, destination)

    return True


def print_statistics(dataset_name, stats):
    print(f"\n{dataset_name} statistics:")

    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")


def normalize_answer(answer):
    """
    Convert valid answers into strings.

    Numeric answers such as 2, 4 and 12 are valid VQA answers.
    Empty and whitespace-only answers are rejected.
    """

    if answer is None:
        return None

    normalized = str(answer).strip()

    if not normalized:
        return None

    return normalized


# ============================================================
# SLAKE PREPROCESSING
# ============================================================

def prepare_slake():

    print("\n" + "=" * 60)
    print("PREPARING SLAKE")
    print("=" * 60)

    source_dir = RAW / "SLAKE" / "Slake1.0"
    image_dir = source_dir / "imgs"

    check_required_path(source_dir)
    check_required_path(image_dir)

    split_files = {
        "train": "train.json",
        "validation": "validate.json",
        "test": "test.json",
    }

    stats = Counter()

    for split, filename in split_files.items():

        annotation_path = source_dir / filename
        check_required_path(annotation_path)

        with annotation_path.open(
            "r", encoding="utf-8"
        ) as file:
            records = json.load(file)

        output_path = ANNOTATIONS / f"slake_{split}.jsonl"

        with output_path.open(
            "w", encoding="utf-8"
        ) as output_file:

            for item in records:

                # Retain English-language questions only.
                if item.get("q_lang") != "en":
                    stats["non_english_skipped"] += 1
                    continue

                question = item.get("question")
                answer = normalize_answer(item.get("answer"))
                image_name = item.get("img_name")

                # Exclude records with missing answers.
                if answer is None:
                    stats["empty_answers_skipped"] += 1
                    continue

                if not question or not image_name:
                    stats["incomplete_records"] += 1
                    continue

                source_image = image_dir / image_name

                destination = IMAGES / "slake" / image_name

                if not copy_image(source_image, destination):
                    stats["missing_images"] += 1
                    continue

                record = {
                    "dataset": "SLAKE",
                    "split": split,
                    "image_path": destination.relative_to(
                        ROOT
                    ).as_posix(),
                    "question": question,
                    "answer": answer,
                    "question_type": item.get("content_type"),
                    "answer_type": item.get("answer_type"),
                    "modality": item.get("modality"),
                    "source_id": item.get("qid"),
                    "source_image_id": item.get("img_id"),
                }

                write_record(record, output_file)
                stats[f"{split}_records"] += 1

    print_statistics("SLAKE", stats)

    return stats


# ============================================================
# VQA-RAD SPLITTING
# ============================================================

def create_vqarad_splits(records):

    """
    Create reproducible 80/10/10 splits.

    All questions belonging to the same image remain
    in the same split to prevent image leakage.
    """

    image_names = sorted({
        item["image_name"]
        for item in records
        if item.get("image_name")
    })

    rng = random.Random(SEED)
    rng.shuffle(image_names)

    total = len(image_names)

    train_end = int(total * 0.80)
    validation_end = train_end + int(total * 0.10)

    assignments = {}

    for image in image_names[:train_end]:
        assignments[image] = "train"

    for image in image_names[train_end:validation_end]:
        assignments[image] = "validation"

    for image in image_names[validation_end:]:
        assignments[image] = "test"

    return assignments


# ============================================================
# VQA-RAD PREPROCESSING
# ============================================================

def prepare_vqarad():

    print("\n" + "=" * 60)
    print("PREPARING VQA-RAD")
    print("=" * 60)

    annotation_path = (
        RAW / "VQA_RAD" / "VQA_RAD Dataset Public.json"
    )

    image_dir = (
        RAW / "VQA_RAD" / "VQA_RAD Image Folder"
    )

    check_required_path(annotation_path)
    check_required_path(image_dir)

    with annotation_path.open(
        "r", encoding="utf-8"
    ) as file:
        records = json.load(file)

    assignments = create_vqarad_splits(records)

    stats = Counter()

    output_files = {
        split: (ANNOTATIONS / f"vqa_rad_{split}.jsonl").open(
            "w", encoding="utf-8"
        )
        for split in ("train", "validation", "test")
    }

    try:

        for item in records:

            image_name = item.get("image_name")
            question = item.get("question")
            answer = normalize_answer(item.get("answer"))

            if answer is None:
                stats["empty_answers_skipped"] += 1
                continue

            if not image_name or not question:
                stats["incomplete_records"] += 1
                continue

            split = assignments.get(image_name)

            if split is None:
                stats["unassigned_records"] += 1
                continue

            safe_name = Path(image_name).name

            source_image = image_dir / safe_name
            destination = IMAGES / "vqa_rad" / safe_name

            if not copy_image(source_image, destination):
                stats["missing_images"] += 1
                continue

            record = {
                "dataset": "VQA_RAD",
                "split": split,
                "image_path": destination.relative_to(
                    ROOT
                ).as_posix(),
                "question": question,
                "answer": answer,
                "question_type": item.get("question_type"),
                "answer_type": item.get("answer_type"),
                "modality": None,
                "source_id": item.get("qid"),
                "source_image_id": image_name,
            }

            write_record(record, output_files[split])
            stats[f"{split}_records"] += 1

    finally:

        for file in output_files.values():
            file.close()

    stats["unique_images"] = len(assignments)

    print_statistics("VQA-RAD", stats)

    return stats


# ============================================================
# PATHVQA PREPROCESSING
# ============================================================

def prepare_pathvqa():

    print("\n" + "=" * 60)
    print("PREPARING PATHVQA")
    print("=" * 60)

    data_dir = RAW / "PathVQA" / "data"

    check_required_path(data_dir)

    split_patterns = {
        "train": "train-*.parquet",
        "validation": "validation-*.parquet",
        "test": "test-*.parquet",
    }

    stats = Counter()

    for split, pattern in split_patterns.items():

        parquet_files = sorted(data_dir.glob(pattern))

        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for PathVQA {split} split."
            )

        output_path = ANNOTATIONS / f"pathvqa_{split}.jsonl"

        with output_path.open(
            "w", encoding="utf-8"
        ) as output_file:

            for parquet_file in parquet_files:

                print(f"Reading: {parquet_file.name}")

                parquet = pq.ParquetFile(parquet_file)

                for batch in parquet.iter_batches(
                    batch_size=PARQUET_BATCH_SIZE
                ):

                    for item in batch.to_pylist():

                        image = item.get("image") or {}

                        image_bytes = image.get("bytes")
                        question = item.get("question")
                        answer = normalize_answer(item.get("answer"))

                        if answer is None:
                            stats["empty_answers_skipped"] += 1
                            continue

                        if not image_bytes or not question:
                            stats["incomplete_records"] += 1
                            continue

                        original_path = image.get("path") or ""

                        suffix = (
                            Path(original_path).suffix or ".jpg"
                        ).lower()

                        # Hash image bytes to create a stable,
                        # collision-resistant filename.
                        image_hash = hashlib.sha256(
                            image_bytes
                        ).hexdigest()

                        filename = f"{image_hash}{suffix}"

                        destination = (
                            IMAGES / "pathvqa" / filename
                        )

                        destination.parent.mkdir(
                            parents=True, exist_ok=True
                        )

                        # Avoid writing the same image repeatedly.
                        if not destination.exists():
                            destination.write_bytes(image_bytes)

                        record = {
                            "dataset": "PathVQA",
                            "split": split,
                            "image_path": destination.relative_to(
                                ROOT
                            ).as_posix(),
                            "question": question,
                            "answer": answer,
                            "question_type": None,
                            "answer_type": None,
                            "modality": "Pathology",
                            "source_id": original_path or image_hash,
                            "source_image_id": original_path or image_hash,
                        }

                        write_record(record, output_file)

                        stats[f"{split}_records"] += 1

    print_statistics("PathVQA", stats)

    return stats


# ============================================================
# FINAL REPORT
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("MEDICAL VQA DATA PREPARATION PIPELINE")
    print("=" * 60)

    prepare_output_directories()

    all_stats = {}

    all_stats["SLAKE"] = prepare_slake()
    all_stats["VQA_RAD"] = prepare_vqarad()
    all_stats["PathVQA"] = prepare_pathvqa()

    report = {
        "project": "Medical-VQA-PEFT-Research",
        "random_seed": SEED,
        "vqarad_split_ratio": {
            "train": 0.80,
            "validation": 0.10,
            "test": 0.10,
        },
        "datasets": {
            name: dict(stats)
            for name, stats in all_stats.items()
        },
    }

    report_path = PROCESSED / "preparation_report.json"

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)

    print(f"\nAnnotations: {ANNOTATIONS}")
    print(f"Images: {IMAGES}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
