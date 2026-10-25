import os
import json
import argparse
import numpy as np
import pandas as pd


def split_patient_dataset(
    input_csv: str = "data/raw/oncomarl_v2_dataset.csv",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Splits an OncoMARL v2 dataset at the patient_id level without data leakage.

    Ensures that every trajectory belonging to a patient is assigned to
    exactly one split (Train, Val, or Test).
    """
    assert np.isclose(
        train_ratio + val_ratio + test_ratio, 1.0
    ), "Split ratios must sum to 1.0!"

    os.makedirs(output_dir, exist_ok=True)
    print(f"[PatientSplit] Loading dataset: '{input_csv}'...")
    df = pd.read_csv(input_csv)

    # Extract sorted unique patient IDs for reproducible shuffling
    patient_ids = np.sort(df["patient_id"].unique())
    total_patients = len(patient_ids)
    total_transitions = len(df)

    print(
        f"[PatientSplit] Found {total_patients} unique patients ({total_transitions} transitions)."
    )

    # Deterministic patient-level shuffling
    rng = np.random.default_rng(seed)
    shuffled_pids = patient_ids.copy()
    rng.shuffle(shuffled_pids)

    # Compute split boundaries
    n_train = int(total_patients * train_ratio)
    n_val = int(total_patients * val_ratio)

    train_pids = set(shuffled_pids[:n_train])
    val_pids = set(shuffled_pids[n_train : n_train + n_val])
    test_pids = set(shuffled_pids[n_train + n_val :])

    # Assert 0 patient overlap across splits (Leakage Prevention Verification)
    assert len(train_pids & val_pids) == 0, "CRITICAL ERROR: Train & Val overlap!"
    assert len(train_pids & test_pids) == 0, (
        "CRITICAL ERROR: Train & Test overlap!"
    )
    assert len(val_pids & test_pids) == 0, "CRITICAL ERROR: Val & Test overlap!"

    # Partition DataFrames
    train_df = df[df["patient_id"].isin(train_pids)].copy()
    val_df = df[df["patient_id"].isin(val_pids)].copy()
    test_df = df[df["patient_id"].isin(test_pids)].copy()

    # Save split CSV files
    train_path = os.path.join(output_dir, "train_patients.csv")
    val_path = os.path.join(output_dir, "val_patients.csv")
    test_path = os.path.join(output_dir, "test_patients.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Generate summary report
    summary = {
        "seed": seed,
        "total_patients": total_patients,
        "total_transitions": total_transitions,
        "splits": {
            "train": {
                "patients": len(train_pids),
                "transitions": len(train_df),
                "patient_percentage": round(
                    len(train_pids) / total_patients * 100, 2
                ),
                "file": train_path,
            },
            "validation": {
                "patients": len(val_pids),
                "transitions": len(val_df),
                "patient_percentage": round(
                    len(val_pids) / total_patients * 100, 2
                ),
                "file": val_path,
            },
            "test": {
                "patients": len(test_pids),
                "transitions": len(test_df),
                "patient_percentage": round(
                    len(test_pids) / total_patients * 100, 2
                ),
                "file": test_path,
            },
        },
        "leakage_check_passed": True,
    }

    summary_path = os.path.join(output_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[PatientSplit] Splitting Complete Successfully!")
    print(
        f"  • Train Set : {len(train_pids):,} patients ({len(train_df):,} transitions) -> {train_path}"
    )
    print(
        f"  • Val Set   : {len(val_pids):,} patients ({len(val_df):,} transitions) -> {val_path}"
    )
    print(
        f"  • Test Set  : {len(test_pids):,} patients ({len(test_df):,} transitions) -> {test_path}"
    )
    print(f"  • Summary   : Saved to {summary_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patient-Level Data Splitting Script for OncoMARL v2"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/oncomarl_v2_dataset.csv",
        help="Path to raw dataset CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save split datasets",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Train patient ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation patient ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test patient ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient shuffling",
    )

    args = parser.parse_args()
    split_patient_dataset(
        input_csv=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )