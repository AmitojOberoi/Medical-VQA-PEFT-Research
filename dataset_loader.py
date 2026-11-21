
"""
dataset_loader.py

Loads processed Medical VQA datasets from JSONL annotations.

Supported datasets:
    - SLAKE
    - VQA_RAD
    - PathVQA

Annotation directory:
    data/processed/annotations/

Image directory:
    data/processed/images/
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

ANNOTATION_DIR = (
    PROJECT_ROOT / "data" / "processed" / "annotations"
)

SUPPORTED_DATASETS = {
    "slake": "SLAKE",
    "vqa_rad": "VQA_RAD",
    "pathvqa": "PathVQA",
}

SUPPORTED_SPLITS = {
    "train",
    "validation",
    "test",
}


# --------------------------------------------------
# MEDICAL VQA DATASET CLASS
# --------------------------------------------------

class MedicalVQADataset(Dataset):
    """
    PyTorch Dataset for Medical VQA annotations.

    Each annotation contains:
        dataset
        split
        image_path
        question
        answer

    Additional metadata is preserved.
    """

    def __init__(
        self,
        dataset_name,
        split="train",
    ):

        # Normalize dataset name without losing
        # its canonical capitalization.

        normalized_name = str(dataset_name).strip().lower()

        if normalized_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported datasets: "
                f"{list(SUPPORTED_DATASETS.values())}"
            )

        self.dataset_name = SUPPORTED_DATASETS[
            normalized_name
        ]

        self.split = str(split).strip().lower()

        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split: {split}. "
                f"Supported splits: {SUPPORTED_SPLITS}"
            )

        # Construct annotation file path.

        annotation_filename = (
            f"{normalized_name}_{self.split}.jsonl"
        )

        self.annotation_file = (
            ANNOTATION_DIR / annotation_filename
        )

        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: "
                f"{self.annotation_file}"
            )

        # Load annotations.

        self.records = self._load_annotations()

        if not self.records:
            raise ValueError(
                f"No records found in "
                f"{self.annotation_file}"
            )

        # Validate annotations.

        self._validate_records()

    # --------------------------------------------------
    # LOAD JSONL ANNOTATIONS
    # --------------------------------------------------

    def _load_annotations(self):
        """Read annotations from a JSONL file."""

        records = []

        with self.annotation_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at line "
                        f"{line_number}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object at "
                        f"line {line_number}"
                    )

                records.append(record)

        return records

    # --------------------------------------------------
    # VALIDATE ANNOTATIONS
    # --------------------------------------------------

    def _validate_records(self):
        """Check required annotation fields and values."""

        required_fields = [
            "dataset",
            "split",
            "image_path",
            "question",
            "answer",
        ]

        for index, record in enumerate(self.records):

            for field in required_fields:

                if field not in record:
                    raise ValueError(
                        f"Record {index} is missing "
                        f"required field: {field}"
                    )

            if (
                str(record["dataset"]).strip().lower()
                != self.dataset_name.lower()
            ):
                raise ValueError(
                    f"Dataset mismatch in record {index}"
                )

            if (
                str(record["split"]).strip().lower()
                != self.split
            ):
                raise ValueError(
                    f"Split mismatch in record {index}"
                )

            if not str(record["question"]).strip():
                raise ValueError(
                    f"Empty question in record {index}"
                )

            if not str(record["answer"]).strip():
                raise ValueError(
                    f"Empty answer in record {index}"
                )

            if not str(record["image_path"]).strip():
                raise ValueError(
                    f"Empty image path in record {index}"
                )

    # --------------------------------------------------
    # DATASET LENGTH
    # --------------------------------------------------

    def __len__(self):
        """Return the number of annotations."""

        return len(self.records)

    # --------------------------------------------------
    # LOAD ONE SAMPLE
    # --------------------------------------------------

    def __getitem__(self, index):
        """
        Return one Medical VQA sample.

        Returns:
            image
            image_path
            question
            answer
            dataset
            split
            metadata
        """

        record = self.records[index]

        # Resolve image path relative to project root.

        image_path = (
            PROJECT_ROOT / record["image_path"]
        ).resolve()

        # Prevent paths from escaping project directory.

        if not image_path.is_relative_to(PROJECT_ROOT):
            raise ValueError(
                f"Image path escapes project directory: "
                f"{image_path}"
            )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Load image safely.

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        # Preserve all optional annotation metadata.

        excluded_fields = {
            "dataset",
            "split",
            "image_path",
            "question",
            "answer",
        }

        metadata = {
            key: value
            for key, value in record.items()
            if key not in excluded_fields
        }

        return {
            "image": image,
            "image_path": str(image_path),
            "question": record["question"],
            "answer": record["answer"],
            "dataset": record["dataset"],
            "split": record["split"],
            "metadata": metadata,
        }


# --------------------------------------------------
# CONVENIENCE DATASET LOADER
# --------------------------------------------------

def load_dataset(dataset_name, split="train"):
    """Load a dataset using its name and split."""

    return MedicalVQADataset(
        dataset_name=dataset_name,
        split=split,
    )


# --------------------------------------------------
# LOAD ALL DATASET SPLITS
# --------------------------------------------------

def print_dataset_summary():
    """Display record counts and sample information."""

    for dataset_key, dataset_name in SUPPORTED_DATASETS.items():

        print("\n" + "=" * 55)
        print(f"DATASET: {dataset_name}")
        print("=" * 55)

        for split in sorted(SUPPORTED_SPLITS):

            dataset = load_dataset(
                dataset_name,
                split,
            )

            print(
                f"{split.capitalize():12}: "
                f"{len(dataset)} records"
            )

        # Inspect one training sample.

        sample = load_dataset(
            dataset_name,
            "train",
        )[0]

        print("\nSample information:")
        print("Question:", sample["question"])
        print("Answer:", sample["answer"])
        print("Image size:", sample["image"].size)
        print("Image mode:", sample["image"].mode)
        print("Metadata:", sample["metadata"])


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("Medical VQA Dataset Loader")
    print_dataset_summary()

    print("\nAll dataset splits loaded successfully.")
