from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"C:\SignBridge_Project")

MODEL_DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_data"
)

SOURCE_SPLIT_PATH = (
    MODEL_DATA_DIR
    / "splits"
    / "dataset_split.csv"
)

OUTPUT_DIR = (
    MODEL_DATA_DIR
    / "splits_signer_independent"
)


# ============================================================
# Signer-independent assignment
# ============================================================

TRAIN_SIGNERS = {2, 6, 7}
VALIDATION_SIGNERS = {5}
TEST_SIGNERS = {1, 3, 8}

EXPECTED_CLASSES = 165


def describe_split(dataframe, split_name):
    split_data = dataframe[
        dataframe["split"] == split_name
    ]

    class_counts = (
        split_data
        .groupby("label_id")
        .size()
    )

    return {
        "samples": int(len(split_data)),
        "classes": int(split_data["label_id"].nunique()),
        "signers": sorted(
            int(value)
            for value in split_data["signer_id"].unique()
        ),
        "signer_counts": {
            str(int(signer)): int(count)
            for signer, count in (
                split_data["signer_id"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "minimum_samples_per_present_class": (
            int(class_counts.min())
            if len(class_counts) > 0
            else 0
        ),
        "maximum_samples_per_present_class": (
            int(class_counts.max())
            if len(class_counts) > 0
            else 0
        ),
    }


def main():
    if not SOURCE_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Source split file was not found:\n"
            f"{SOURCE_SPLIT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe = pd.read_csv(
        SOURCE_SPLIT_PATH
    )

    required_columns = {
        "sample_index",
        "label_id",
        "label",
        "signer_id",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["signer_id"] = (
        dataframe["signer_id"].astype(int)
    )

    dataframe["sample_index"] = (
        dataframe["sample_index"].astype(int)
    )

    dataframe["label_id"] = (
        dataframe["label_id"].astype(int)
    )

    all_expected_signers = (
        TRAIN_SIGNERS
        | VALIDATION_SIGNERS
        | TEST_SIGNERS
    )

    dataset_signers = set(
        dataframe["signer_id"].unique()
    )

    unknown_signers = (
        dataset_signers
        - all_expected_signers
    )

    missing_signers = (
        all_expected_signers
        - dataset_signers
    )

    if unknown_signers:
        raise ValueError(
            "Unassigned signer IDs found: "
            + str(sorted(unknown_signers))
        )

    if missing_signers:
        raise ValueError(
            "Expected signer IDs are missing: "
            + str(sorted(missing_signers))
        )

    # Assign each video according to its signer
    dataframe["split"] = ""

    dataframe.loc[
        dataframe["signer_id"].isin(TRAIN_SIGNERS),
        "split"
    ] = "train"

    dataframe.loc[
        dataframe["signer_id"].isin(
            VALIDATION_SIGNERS
        ),
        "split"
    ] = "validation"

    dataframe.loc[
        dataframe["signer_id"].isin(TEST_SIGNERS),
        "split"
    ] = "test"

    if (dataframe["split"] == "").any():
        unassigned = dataframe[
            dataframe["split"] == ""
        ]

        raise ValueError(
            f"{len(unassigned)} samples were not assigned."
        )

    train_data = dataframe[
        dataframe["split"] == "train"
    ]

    validation_data = dataframe[
        dataframe["split"] == "validation"
    ]

    test_data = dataframe[
        dataframe["split"] == "test"
    ]

    train_indices = (
        train_data["sample_index"]
        .to_numpy(dtype=np.int64)
    )

    validation_indices = (
        validation_data["sample_index"]
        .to_numpy(dtype=np.int64)
    )

    test_indices = (
        test_data["sample_index"]
        .to_numpy(dtype=np.int64)
    )

    # ========================================================
    # Safety checks
    # ========================================================

    train_index_set = set(train_indices.tolist())
    validation_index_set = set(
        validation_indices.tolist()
    )
    test_index_set = set(test_indices.tolist())

    no_index_overlap = (
        train_index_set.isdisjoint(
            validation_index_set
        )
        and train_index_set.isdisjoint(
            test_index_set
        )
        and validation_index_set.isdisjoint(
            test_index_set
        )
    )

    all_indices = (
        train_index_set
        | validation_index_set
        | test_index_set
    )

    all_samples_assigned_once = (
        no_index_overlap
        and len(all_indices) == len(dataframe)
        and all_indices
        == set(
            dataframe["sample_index"].tolist()
        )
    )

    train_signer_set = set(
        train_data["signer_id"].unique()
    )

    validation_signer_set = set(
        validation_data["signer_id"].unique()
    )

    test_signer_set = set(
        test_data["signer_id"].unique()
    )

    signer_sets_disjoint = (
        train_signer_set.isdisjoint(
            validation_signer_set
        )
        and train_signer_set.isdisjoint(
            test_signer_set
        )
        and validation_signer_set.isdisjoint(
            test_signer_set
        )
    )

    all_classes = set(
        dataframe["label_id"].unique()
    )

    train_classes = set(
        train_data["label_id"].unique()
    )

    test_classes = set(
        test_data["label_id"].unique()
    )

    all_classes_in_train = (
        train_classes == all_classes
    )

    all_classes_in_test = (
        test_classes == all_classes
    )

    train_class_counts = (
        train_data
        .groupby("label_id")
        .size()
    )

    minimum_train_samples = int(
        train_class_counts.min()
    )

    if len(all_classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(all_classes)}."
        )

    if not all_samples_assigned_once:
        raise ValueError(
            "Sample assignment check failed."
        )

    if not signer_sets_disjoint:
        raise ValueError(
            "Signer overlap was detected."
        )

    if not all_classes_in_train:
        missing_train_classes = sorted(
            all_classes - train_classes
        )

        raise ValueError(
            "Some classes are missing from Train: "
            + str(missing_train_classes)
        )

    if not all_classes_in_test:
        missing_test_classes = sorted(
            all_classes - test_classes
        )

        raise ValueError(
            "Some classes are missing from Test: "
            + str(missing_test_classes)
        )

    if minimum_train_samples < 2:
        raise ValueError(
            "At least one class has fewer than "
            "two training samples."
        )

    # ========================================================
    # Save outputs
    # ========================================================

    np.save(
        OUTPUT_DIR / "train_indices.npy",
        train_indices
    )

    np.save(
        OUTPUT_DIR / "validation_indices.npy",
        validation_indices
    )

    np.save(
        OUTPUT_DIR / "test_indices.npy",
        test_indices
    )

    dataframe.to_csv(
        OUTPUT_DIR
        / "dataset_split_signer_independent.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "split_strategy": (
            "Signer-independent train, validation, and test"
        ),
        "total_samples": int(len(dataframe)),
        "total_classes": int(len(all_classes)),
        "train": describe_split(
            dataframe,
            "train"
        ),
        "validation": describe_split(
            dataframe,
            "validation"
        ),
        "test": describe_split(
            dataframe,
            "test"
        ),
        "checks": {
            "all_samples_assigned_once": (
                all_samples_assigned_once
            ),
            "no_index_overlap": no_index_overlap,
            "signer_sets_disjoint": (
                signer_sets_disjoint
            ),
            "all_classes_in_train": (
                all_classes_in_train
            ),
            "all_classes_in_test": (
                all_classes_in_test
            ),
            "minimum_training_samples_per_class": (
                minimum_train_samples
            ),
        },
        "important_note": (
            "Validation contains only the classes "
            "performed by signer 05. Metrics must be "
            "computed over the classes present in validation."
        ),
    }

    with open(
        OUTPUT_DIR / "split_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 60)
    print("Signer-independent split completed.")
    print(f"Total samples: {len(dataframe)}")
    print(f"Total classes: {len(all_classes)}")
    print()

    print(
        f"Train: {len(train_data)} samples, "
        f"{train_data['label_id'].nunique()} classes, "
        f"signers: {sorted(train_signer_set)}"
    )

    print(
        f"Validation: {len(validation_data)} samples, "
        f"{validation_data['label_id'].nunique()} classes, "
        f"signers: {sorted(validation_signer_set)}"
    )

    print(
        f"Test: {len(test_data)} samples, "
        f"{test_data['label_id'].nunique()} classes, "
        f"signers: {sorted(test_signer_set)}"
    )

    print()
    print(
        "Minimum training samples per class: "
        f"{minimum_train_samples}"
    )

    print(
        "Signer overlap: "
        f"{not signer_sets_disjoint}"
    )

    print(
        "All classes in Train: "
        f"{all_classes_in_train}"
    )

    print(
        "All classes in Test: "
        f"{all_classes_in_test}"
    )

    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()