
import json
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS = ROOT / "data" / "processed" / "annotations"
REPORT_DIR = ROOT / "data" / "processed"

OUTPUT_FILE = REPORT_DIR / "duplicate_analysis_report.json"

MAX_EXAMPLES = 10


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for duplicate detection.

    - Convert to lowercase.
    - Remove punctuation.
    - Remove extra whitespace.
    """

    if value is None:
        return ""

    value = str(value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# LOAD JSONL FILES
# ============================================================

def load_records(file_path):

    records = []

    with file_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path.name}, "
                    f"line {line_number}: {error}"
                )

            records.append(record)

    return records


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_records(records):

    exact_groups = defaultdict(list)
    image_question_groups = defaultdict(list)
    question_image_groups = defaultdict(set)

    for index, record in enumerate(records):

        dataset = record.get("dataset", "")
        image_path = record.get("image_path", "")

        question = normalize_text(
            record.get("question")
        )

        answer = normalize_text(
            record.get("answer")
        )

        # Same dataset + image + question + answer.
        exact_key = (
            dataset,
            image_path,
            question,
            answer,
        )

        # Same dataset + image + question.
        image_question_key = (
            dataset,
            image_path,
            question,
        )

        exact_groups[exact_key].append(index)

        image_question_groups[image_question_key].append(index)

        # Track which different images use the same question.
        question_image_groups[question].add(image_path)

    # Only groups containing more than one record are duplicates.
    exact_duplicates = {
        key: indices
        for key, indices in exact_groups.items()
        if len(indices) > 1
    }

    repeated_image_questions = {
        key: indices
        for key, indices in image_question_groups.items()
        if len(indices) > 1
    }

    repeated_questions_across_images = {
        question: images
        for question, images in question_image_groups.items()
        if question and len(images) > 1
    }

    # Count extra occurrences beyond the first record.
    exact_duplicate_records = sum(
        len(indices) - 1
        for indices in exact_duplicates.values()
    )

    duplicate_image_question_records = sum(
        len(indices) - 1
        for indices in repeated_image_questions.values()
    )

    exact_examples = []

    for key, indices in list(exact_duplicates.items())[:MAX_EXAMPLES]:

        dataset, image_path, question, answer = key

        exact_examples.append({
            "dataset": dataset,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "record_indices": indices,
            "occurrences": len(indices),
        })

    image_question_examples = []

    for key, indices in list(
        repeated_image_questions.items()
    )[:MAX_EXAMPLES]:

        dataset, image_path, question = key

        image_question_examples.append({
            "dataset": dataset,
            "image_path": image_path,
            "question": question,
            "record_indices": indices,
            "occurrences": len(indices),
        })

    cross_image_examples = []

    for question, images in list(
        repeated_questions_across_images.items()
    )[:MAX_EXAMPLES]:

        cross_image_examples.append({
            "question": question,
            "unique_images": len(images),
            "image_paths": sorted(images)[:MAX_EXAMPLES],
        })

    return {
        "total_records": len(records),

        "exact_duplicate_groups": len(exact_duplicates),

        "exact_duplicate_records": exact_duplicate_records,

        "duplicate_image_question_groups": len(
            repeated_image_questions
        ),

        "duplicate_image_question_records": (
            duplicate_image_question_records
        ),

        "repeated_questions_across_images": len(
            repeated_questions_across_images
        ),

        "exact_duplicate_examples": exact_examples,

        "duplicate_image_question_examples": (
            image_question_examples
        ),

        "repeated_question_examples": cross_image_examples,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("MEDICAL VQA DUPLICATE DETECTION")
    print("=" * 65)

    if not ANNOTATIONS.exists():

        raise FileNotFoundError(
            f"Annotation directory not found: {ANNOTATIONS}"
        )

    annotation_files = sorted(
        ANNOTATIONS.glob("*.jsonl")
    )

    if not annotation_files:

        raise FileNotFoundError(
            "No JSONL annotation files found."
        )

    report = {
        "project": "Medical-VQA-PEFT-Research",
        "normalization": (
            "Lowercase, punctuation removal and whitespace normalization"
        ),
        "datasets": {},
        "overall": {},
    }

    overall_records = 0
    overall_exact_duplicates = 0
    overall_image_question_duplicates = 0

    for file_path in annotation_files:

        print("\n" + "-" * 65)
        print(f"Analyzing: {file_path.name}")
        print("-" * 65)

        records = load_records(file_path)

        results = analyze_records(records)

        report["datasets"][file_path.stem] = results

        overall_records += results["total_records"]

        overall_exact_duplicates += (
            results["exact_duplicate_records"]
        )

        overall_image_question_duplicates += (
            results["duplicate_image_question_records"]
        )

        print(f"Total records: {results['total_records']}")

        print(
            "Exact duplicate groups:",
            results["exact_duplicate_groups"]
        )

        print(
            "Exact duplicate records:",
            results["exact_duplicate_records"]
        )

        print(
            "Duplicate image-question groups:",
            results["duplicate_image_question_groups"]
        )

        print(
            "Duplicate image-question records:",
            results["duplicate_image_question_records"]
        )

        print(
            "Repeated questions across different images:",
            results["repeated_questions_across_images"]
        )

    report["overall"] = {
        "total_records_analyzed": overall_records,
        "exact_duplicate_records": overall_exact_duplicates,
        "duplicate_image_question_records": (
            overall_image_question_duplicates
        ),
        "files_analyzed": len(annotation_files),
    }

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
    print("DUPLICATE ANALYSIS SUMMARY")
    print("=" * 65)

    print(f"Files analyzed: {len(annotation_files)}")
    print(f"Records analyzed: {overall_records}")

    print(
        "Exact duplicate records:",
        overall_exact_duplicates
    )

    print(
        "Duplicate image-question records:",
        overall_image_question_duplicates
    )

    print(f"\nReport saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
