
import json

import pytest
from PIL import Image

import dataset


@pytest.fixture
def sample_project(tmp_path, monkeypatch):
    """Create a temporary project with one valid medical image."""

    monkeypatch.setattr(dataset, "PROJECT_ROOT", tmp_path)

    annotation_dir = tmp_path / "data" / "processed" / "annotations"
    image_dir = tmp_path / "data" / "processed" / "images"

    annotation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    Image.new("L", (64, 64), color=128).save(
        image_dir / "sample.png"
    )

    record = {
        "dataset": "SLAKE",
        "split": "train",
        "image_path": "data/processed/images/sample.png",
        "question": "What modality is used?",
        "answer": "MRI",
        "question_type": "Modality",
        "answer_type": "OPEN",
        "modality": "MRI",
        "source_id": 1,
        "source_image_id": 1,
    }

    annotation_file = annotation_dir / "slake_train.jsonl"
    annotation_file.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    return tmp_path, annotation_file, record


def test_load_valid_dataset(sample_project):
    data = dataset.MedicalVQADataset("SLAKE", "train")
    assert len(data) == 1


def test_dataset_name_is_case_insensitive(sample_project):
    data = dataset.MedicalVQADataset("slake", "train")
    assert data.dataset_name == "SLAKE"


def test_all_supported_datasets():
    assert "SLAKE" in dataset.DATASET_PREFIXES
    assert "VQA_RAD" in dataset.DATASET_PREFIXES
    assert "PATHVQA" in dataset.DATASET_PREFIXES


def test_invalid_dataset_raises_error(sample_project):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        dataset.MedicalVQADataset("INVALID", "train")


def test_invalid_split_raises_error(sample_project):
    with pytest.raises(ValueError, match="Invalid split"):
        dataset.MedicalVQADataset("SLAKE", "invalid")


def test_missing_annotation_file_raises_error(sample_project):
    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        dataset.MedicalVQADataset("VQA_RAD", "train")


def test_invalid_json_raises_error(sample_project):
    _, annotation_file, _ = sample_project
    annotation_file.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        dataset.MedicalVQADataset("SLAKE", "train")


def test_missing_required_field_raises_error(sample_project):
    _, annotation_file, _ = sample_project
    annotation_file.write_text(
        json.dumps({"dataset": "SLAKE"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing fields"):
        dataset.MedicalVQADataset("SLAKE", "train")


def test_dataset_mismatch_raises_error(sample_project):
    _, annotation_file, record = sample_project
    record["dataset"] = "VQA_RAD"
    annotation_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset mismatch"):
        dataset.MedicalVQADataset("SLAKE", "train")


def test_split_mismatch_raises_error(sample_project):
    _, annotation_file, record = sample_project
    record["split"] = "test"
    annotation_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Split mismatch"):
        dataset.MedicalVQADataset("SLAKE", "train")


@pytest.mark.parametrize("field", ["question", "answer"])
def test_empty_text_fields_are_rejected(sample_project, field):
    _, annotation_file, record = sample_project
    record[field] = "   "
    annotation_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    message = "Empty question" if field == "question" else "Empty answer"

    with pytest.raises(ValueError, match=message):
        dataset.MedicalVQADataset("SLAKE", "train")


def test_empty_dataset_raises_error(sample_project):
    _, annotation_file, _ = sample_project
    annotation_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No records found"):
        dataset.MedicalVQADataset("SLAKE", "train")


def test_max_samples_limits_dataset(sample_project):
    _, annotation_file, record = sample_project
    annotation_file.write_text(
        (json.dumps(record) + "\n") * 3,
        encoding="utf-8",
    )

    data = dataset.MedicalVQADataset(
        "SLAKE", "train", max_samples=2
    )
    assert len(data) == 2


def test_invalid_max_samples_raises_error(sample_project):
    with pytest.raises(ValueError, match="max_samples must be positive"):
        dataset.MedicalVQADataset(
            "SLAKE", "train", max_samples=0
        )


def test_sample_contains_expected_fields(sample_project):
    data = dataset.MedicalVQADataset("SLAKE", "train")
    sample = data[0]

    assert sample["question"] == "What modality is used?"
    assert sample["answer"] == "MRI"
    assert sample["dataset"] == "SLAKE"
    assert sample["split"] == "train"
    assert sample["question_type"] == "Modality"


def test_image_is_converted_to_rgb(sample_project):
    data = dataset.MedicalVQADataset("SLAKE", "train")
    sample = data[0]

    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].mode == "RGB"
    assert sample["image"].size == (64, 64)


def test_missing_image_raises_error(sample_project):
    _, annotation_file, record = sample_project
    record["image_path"] = "data/processed/images/missing.png"
    annotation_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    data = dataset.MedicalVQADataset("SLAKE", "train")

    with pytest.raises(FileNotFoundError, match="Medical image not found"):
        data[0]


def test_absolute_image_path_is_rejected(sample_project):
    _, annotation_file, record = sample_project
    record["image_path"] = "C:/outside/image.png"
    annotation_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    data = dataset.MedicalVQADataset("SLAKE", "train")

    with pytest.raises(ValueError, match="Absolute image paths"):
        data[0]


def test_path_traversal_is_rejected(sample_project):
    _, annotation_file, record = sample_project
    record["image_path"] = "../../outside.png"
    annotation_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    data = dataset.MedicalVQADataset("SLAKE", "train")

    with pytest.raises(ValueError, match="escapes project directory"):
        data[0]


def test_collate_function(sample_project):
    data = dataset.MedicalVQADataset("SLAKE", "train")
    batch = dataset.medical_vqa_collate_fn([data[0]])

    assert len(batch["images"]) == 1
    assert batch["questions"] == ["What modality is used?"]
    assert batch["answers"] == ["MRI"]
    assert batch["datasets"] == ["SLAKE"]


def test_build_dataloader(sample_project):
    loader = dataset.build_dataloader(
        "SLAKE",
        "train",
        batch_size=1,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert len(batch["images"]) == 1
    assert batch["answers"] == ["MRI"]


def test_invalid_batch_size_raises_error(sample_project):
    with pytest.raises(ValueError, match="batch_size"):
        dataset.build_dataloader("SLAKE", "train", batch_size=0)


def test_invalid_num_workers_raises_error(sample_project):
    with pytest.raises(ValueError, match="num_workers"):
        dataset.build_dataloader("SLAKE", "train", num_workers=-1)
