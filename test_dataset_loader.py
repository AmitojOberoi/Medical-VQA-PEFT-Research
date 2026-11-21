

from dataset import MedicalVQADataset, build_dataloader


def main():

    datasets = ["SLAKE", "VQA_RAD", "PathVQA"]
    splits = ["train", "validation", "test"]

    total_records = 0

    for dataset_name in datasets:

        for split in splits:

            print("\n" + "=" * 60)
            print(f"TESTING {dataset_name} - {split}")
            print("=" * 60)

            dataset = MedicalVQADataset(
                dataset_name=dataset_name,
                split=split,
            )

            sample = dataset[0]

            assert sample["image"].mode == "RGB"
            assert sample["question"].strip()
            assert sample["answer"].strip()

            loader = build_dataloader(
                dataset_name=dataset_name,
                split=split,
                batch_size=2,
            )

            batch = next(iter(loader))

            assert len(batch["images"]) == 2
            assert len(batch["questions"]) == 2
            assert len(batch["answers"]) == 2

            print("Total records:", len(dataset))
            print("Image size:", sample["image"].size)
            print("Sample question:", sample["question"])
            print("Sample answer:", sample["answer"])
            print("Batch size:", len(batch["images"]))
            print("STATUS: PASS")

            total_records += len(dataset)

    print("\n" + "=" * 60)
    print("ALL DATASET TESTS COMPLETED")
    print("Total records:", total_records)
    print("Dataset/split combinations:", len(datasets) * len(splits))
    print("=" * 60)


if __name__ == "__main__":
    main()
