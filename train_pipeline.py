
"""
train_pipeline.py

Prepares Medical VQA annotations for LLaVA-style
multimodal fine-tuning.

Input:
    data/processed/annotations/*.jsonl

Output:
    data/processed/llava_format/*.json
"""

import argparse
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

DATASET_PREFIXES = {
    "SLAKE": "slake",
    "VQA_RAD": "vqa_rad",
    "PathVQA": "pathvqa",
}

SPLITS = ("train", "validation", "test")


def load_config():
    """Load and validate the project configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid YAML configuration.")

    for section in ("project", "paths", "datasets"):
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    return config


def resolve_project_path(path_value):
    """Resolve a configured path relative to the project root."""

    path = Path(path_value)

    if path.is_absolute():
        raise ValueError(
            f"Configured paths must be project-relative: {path}"
        )

    resolved = (PROJECT_ROOT / path).resolve()

    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path escapes project directory: {path}"
        ) from error

    return resolved


def get_annotation_path(dataset, split):
    """Return the JSONL annotation file for a dataset split."""

    prefix = DATASET_PREFIXES[dataset]

    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "annotations"
        / f"{prefix}_{split}.jsonl"
    )


def load_records(annotation_path, expected_dataset, expected_split):
    """Read and validate JSONL annotation records."""

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    records = []

    required_fields = {
        "dataset",
        "split",
        "image_path",
        "question",
        "answer",
    }

    with annotation_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in {annotation_path.name}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_number}"
                )

            missing = required_fields - record.keys()

            if missing:
                raise ValueError(
                    f"Missing fields at line {line_number}: "
                    f"{sorted(missing)}"
                )

            if record["dataset"] != expected_dataset:
                raise ValueError(
                    f"Dataset mismatch at line {line_number}"
                )

            if record["split"] != expected_split:
                raise ValueError(
                    f"Split mismatch at line {line_number}"
                )

            if not str(record["question"]).strip():
                raise ValueError(
                    f"Empty question at line {line_number}"
                )

            if not str(record["answer"]).strip():
                raise ValueError(
                    f"Empty answer at line {line_number}"
                )

            records.append(record)

    return records


def convert_record(record, expected_dataset, expected_split):
    """
    Convert one annotation into the LLaVA conversation format.
    """

    relative_image_path = Path(record["image_path"])

    if relative_image_path.is_absolute():
        raise ValueError(
            f"Absolute image path is not allowed: "
            f"{relative_image_path}"
        )

    image_path = resolve_project_path(
        relative_image_path
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    return {
        "id": str(
            record.get(
                "source_id",
                f"{expected_dataset}_{expected_split}",
            )
        ),
        "image": relative_image_path.as_posix(),
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\n"
                    + str(record["question"]).strip()
                ),
            },
            {
                "from": "gpt",
                "value": str(record["answer"]).strip(),
            },
        ],
        "metadata": {
            key: value
            for key, value in record.items()
            if key not in {
                "image_path",
                "question",
                "answer",
                "dataset",
                "split",
                "source_id",
            }
        },
    }


def prepare_split(
    dataset,
    split,
    output_dir,
    limit=None,
):
    """Convert and save one dataset split."""

    annotation_path = get_annotation_path(
        dataset,
        split,
    )

    records = load_records(
        annotation_path,
        dataset,
        split,
    )

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")

        records = records[:limit]

    converted_records = [
        convert_record(record, dataset, split)
        for record in records
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{DATASET_PREFIXES[dataset]}_{split}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            converted_records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"{dataset:10} | {split:10} | "
        f"{len(converted_records):6} records | "
        f"Saved: {output_path.relative_to(PROJECT_ROOT)}"
    )

    return len(converted_records)


def main():

    parser = argparse.ArgumentParser(
        description="Prepare Medical VQA data for LLaVA."
    )

    parser.add_argument(
        "--dataset",
        choices=["all", "SLAKE", "VQA_RAD", "PathVQA"],
        default="all",
    )

    parser.add_argument(
        "--split",
        choices=["all", "train", "validation", "test"],
        default="all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config = load_config()

    # Use the explicit llava_format path.
    output_dir = resolve_project_path(
        config["paths"].get(
            "llava_format",
            "data/processed/llava_format",
        )
    )

    datasets = (
        list(DATASET_PREFIXES)
        if args.dataset == "all"
        else [args.dataset]
    )

    splits = (
        list(SPLITS)
        if args.split == "all"
        else [args.split]
    )

    total_records = 0

    print("\nMedical VQA training-data preparation")
    print("-------------------------------------")

    for dataset in datasets:

        dataset_config = config["datasets"].get(dataset)

        if dataset_config is None:
            raise ValueError(
                f"{dataset} is missing from config.yaml"
            )

        if not dataset_config.get("enabled", False):
            print(f"Skipping disabled dataset: {dataset}")
            continue

        for split in splits:

            count = prepare_split(
                dataset=dataset,
                split=split,
                output_dir=output_dir,
                limit=args.limit,
            )

            total_records += count

    print("-------------------------------------")
    print(f"Total records prepared: {total_records}")
    print(f"Output directory: {output_dir}")
    print("\nData preparation completed successfully.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()
