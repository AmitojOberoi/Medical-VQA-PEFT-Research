
"""
validate_llava_data.py

Validates the converted LLaVA-format Medical VQA data:
- All expected dataset/split files exist.
- JSON structure is valid.
- Conversation format is correct.
- Referenced images exist.
- Record counts match the prepared datasets.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "llava_format"

EXPECTED_COUNTS = {
    "slake_train.json": 4918,
    "slake_validation.json": 1053,
    "slake_test.json": 1061,
    "vqa_rad_train.json": 1791,
    "vqa_rad_validation.json": 216,
    "vqa_rad_test.json": 241,
    "pathvqa_train.json": 19648,
    "pathvqa_validation.json": 6259,
    "pathvqa_test.json": 6719,
}


def validate_file(filename, expected_count):
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{filename}: expected a JSON list.")

    if len(records) != expected_count:
        raise ValueError(
            f"{filename}: expected {expected_count} records, "
            f"found {len(records)}."
        )

    for index, record in enumerate(records):
        required = {"id", "image", "conversations", "metadata"}

        if not required.issubset(record):
            raise ValueError(
                f"{filename}, record {index}: missing fields."
            )

        image_path = (PROJECT_ROOT / record["image"]).resolve()

        try:
            image_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"{filename}, record {index}: image path escapes project."
            ) from error

        if not image_path.is_file():
            raise FileNotFoundError(
                f"{filename}, record {index}: image missing: {image_path}"
            )

        conversations = record["conversations"]

        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(
                f"{filename}, record {index}: invalid conversations."
            )

        human, assistant = conversations

        if human.get("from") != "human":
            raise ValueError(
                f"{filename}, record {index}: first speaker must be human."
            )

        if assistant.get("from") != "gpt":
            raise ValueError(
                f"{filename}, record {index}: second speaker must be gpt."
            )

        if "<image>" not in human.get("value", ""):
            raise ValueError(
                f"{filename}, record {index}: missing image token."
            )

        if not assistant.get("value", "").strip():
            raise ValueError(
                f"{filename}, record {index}: empty answer."
            )

    return len(records)


def main():
    print("\nLLaVA-format dataset validation")
    print("--------------------------------")

    total = 0

    for filename, expected_count in EXPECTED_COUNTS.items():
        count = validate_file(filename, expected_count)
        total += count

        print(
            f"PASS | {filename:30} | {count:6} records"
        )

    print("--------------------------------")
    print(f"Total validated records: {total}")
    print("All files and image references passed validation.")


if __name__ == "__main__":
    main()
