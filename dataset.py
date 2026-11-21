
"""
dataset.py

PyTorch Dataset and DataLoader implementation
for the Medical VQA research project.

Supported datasets:
    - SLAKE
    - VQA_RAD
    - PathVQA
"""

import json
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader

from utils import PROJECT_ROOT


# --------------------------------------------------
# 1. DATASET CONFIGURATION
# --------------------------------------------------

DATASET_PREFIXES = {
    "SLAKE": {
        "prefix": "slake",
        "label": "SLAKE",
    },
    "VQA_RAD": {
        "prefix": "vqa_rad",
        "label": "VQA_RAD",
    },
    "PATHVQA": {
        "prefix": "pathvqa",
        "label": "PathVQA",
    },
}

VALID_SPLITS = ("train", "validation", "test")


# --------------------------------------------------
# 2. MEDICAL VQA DATASET
# --------------------------------------------------

class MedicalVQADataset(Dataset):

    def __init__(
        self,
        dataset_name: str,
        split: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):

        dataset_key = dataset_name.strip().upper()
        self.split = split.strip().lower()
        self.transform = transform

        # Validate dataset name
        if dataset_key not in DATASET_PREFIXES:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Choose from SLAKE, VQA_RAD, PathVQA."
            )

        self.dataset_name = DATASET_PREFIXES[dataset_key]["label"]
        prefix = DATASET_PREFIXES[dataset_key]["prefix"]

        # Validate split
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {self.split}. "
                f"Choose from {VALID_SPLITS}."
            )

        # Locate annotation file
        self.annotation_file = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "annotations"
            / f"{prefix}_{self.split}.jsonl"
        )

        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        # Load annotations
        self.records = self._load_annotations()

        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive.")

            self.records = self.records[:max_samples]

        if not self.records:
            raise ValueError(
                f"No records found for "
                f"{self.dataset_name} ({self.split})."
            )

        print(
            f"Loaded {len(self.records)} records: "
            f"{self.dataset_name} / {self.split}"
        )

    # --------------------------------------------------
    # 3. LOAD JSONL ANNOTATIONS
    # --------------------------------------------------

    def _load_annotations(self):

        records = []

        with self.annotation_file.open(
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
                        f"Invalid JSON in "
                        f"{self.annotation_file.name}, "
                        f"line {line_number}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Record at line {line_number} "
                        f"must be a JSON object."
                    )

                required_fields = {
                    "dataset",
                    "split",
                    "image_path",
                    "question",
                    "answer",
                }

                missing = required_fields - record.keys()

                if missing:
                    raise ValueError(
                        f"Missing fields at line {line_number}: "
                        f"{sorted(missing)}"
                    )

                if record["dataset"] != self.dataset_name:
                    raise ValueError(
                        f"Dataset mismatch at line {line_number}: "
                        f"expected {self.dataset_name}, "
                        f"got {record['dataset']}."
                    )

                if record["split"] != self.split:
                    raise ValueError(
                        f"Split mismatch at line {line_number}: "
                        f"expected {self.split}, "
                        f"got {record['split']}."
                    )

                if not str(record["question"]).strip():
                    raise ValueError(
                        f"Empty question at line {line_number}."
                    )

                if not str(record["answer"]).strip():
                    raise ValueError(
                        f"Empty answer at line {line_number}."
                    )

                records.append(record)

        return records

    # --------------------------------------------------
    # 4. DATASET LENGTH
    # --------------------------------------------------

    def __len__(self):
        return len(self.records)

    # --------------------------------------------------
    # 5. LOAD INDIVIDUAL SAMPLE
    # --------------------------------------------------

    def __getitem__(self, index):

        record = self.records[index]

        relative_image_path = Path(record["image_path"])

        if relative_image_path.is_absolute():
            raise ValueError(
                "Absolute image paths are not allowed."
            )

        image_path = (
            PROJECT_ROOT / relative_image_path
        ).resolve()

        # Prevent paths from escaping the project directory
        try:
            image_path.relative_to(PROJECT_ROOT.resolve())

        except ValueError as error:
            raise ValueError(
                f"Image path escapes project directory: "
                f"{record['image_path']}"
            ) from error

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Medical image not found: {image_path}"
            )

        # Open and convert image to RGB
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:

            raise RuntimeError(
                f"Unable to load image: {image_path}"
            ) from error

        # Apply optional image transformation
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "question": str(record["question"]),
            "answer": str(record["answer"]),
            "dataset": self.dataset_name,
            "split": self.split,
            "question_type": record.get("question_type"),
            "answer_type": record.get("answer_type"),
            "modality": record.get("modality"),
            "source_id": record.get("source_id"),
            "source_image_id": record.get("source_image_id"),
        }


# --------------------------------------------------
# 6. CUSTOM BATCH COLLATION
# --------------------------------------------------

def medical_vqa_collate_fn(batch):

    return {
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "datasets": [item["dataset"] for item in batch],
        "splits": [item["split"] for item in batch],
        "question_types": [
            item["question_type"] for item in batch
        ],
        "answer_types": [
            item["answer_type"] for item in batch
        ],
        "modalities": [
            item["modality"] for item in batch
        ],
        "source_ids": [
            item["source_id"] for item in batch
        ],
        "source_image_ids": [
            item["source_image_id"] for item in batch
        ],
    }


# --------------------------------------------------
# 7. DATALOADER FACTORY
# --------------------------------------------------

def build_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    max_samples: Optional[int] = None,
):

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if shuffle is None:
        shuffle = split.strip().lower() == "train"

    dataset = MedicalVQADataset(
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=medical_vqa_collate_fn,
        pin_memory=False,
    )
