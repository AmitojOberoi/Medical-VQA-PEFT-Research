
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def inspect_json(path):
    print(f"\nFILE: {path.name}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("Type:", type(data).__name__)

    if isinstance(data, list):
        print("Number of records:", len(data))
        for record in data[:2]:
            print(json.dumps(record, indent=2, ensure_ascii=False))

    elif isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])


def inspect_slake():
    print("\n========== SLAKE ==========")
    folder = RAW / "SLAKE" / "Slake1.0"

    for filename in ["train.json", "validate.json", "test.json"]:
        inspect_json(folder / filename)


def inspect_vqarad():
    print("\n========== VQA-RAD ==========")
    path = RAW / "VQA_RAD" / "VQA_RAD Dataset Public.json"
    inspect_json(path)


def inspect_pathvqa():
    print("\n========== PathVQA ==========")
    folder = RAW / "PathVQA" / "data"

    parquet_files = sorted(folder.glob("train-*.parquet"))

    if not parquet_files:
        print("No training Parquet files found!")
        return

    path = parquet_files[0]
    parquet = pq.ParquetFile(path)

    print("File:", path.name)
    print("Columns and schema:")
    print(parquet.schema)

    print("\nFirst two records:")
    table = pq.read_table(path).slice(0, 2)

    for record in table.to_pylist():
        # Avoid printing the entire embedded image bytes.
        for key, value in record.items():
            if isinstance(value, dict) and "bytes" in value:
                value = {
                    "image_bytes_present": value["bytes"] is not None,
                    "image_path": value.get("path")
                }
            record[key] = value

        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    inspect_slake()
    inspect_vqarad()
    inspect_pathvqa()
