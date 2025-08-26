#!/usr/bin/env python3
"""
make_camelyon16_folds.py
Create ACMIL-style 5 splits for CAMELYON16:
- Fixed official test (same for all folds) from split == "test"
- Stratified 90/10 train/val inside the official train set
- Outputs: fold_1..fold_5 with train.csv, val.csv, test.csv


python generate_camelyon16_split_cross_validation.py --csv ./dataset_csv/camelyon16.csv --out_root ./dataset_csv/camelyon16_folds

"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    p = argparse.ArgumentParser(description="Create ACMIL-style 5 splits for CAMELYON16")
    p.add_argument("--csv", type=str, required=True,
                   help="Path to camelyon16.csv with columns: slide_id, full_path, label, split")
    p.add_argument("--out_root", type=str, required=True,
                   help="Output directory (will create fold_1..fold_5 subfolders)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="Random seeds for the 5 folds (default: 42 43 44 45 46)")
    p.add_argument("--test_size", type=float, default=0.1,
                   help="Validation fraction from official train (default: 0.1)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing fold files if present")
    return p.parse_args()


def ensure_cols(df, needed):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. "
                         f"Found: {list(df.columns)}")


def normalize_label_series(s):
    """
    Robust label normalization:
    - If numeric, cast to int (assumes 0/1 already).
    - If text, map IDC->1, ILC/Normal->0 for generality; but CAMELYON typically uses 0/1 already.
      Here we keep it simple: if text and {0,1} not parseable, try common tokens.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)

    st = s.astype(str).str.strip().str.upper()
    # Common tokens (feel free to tweak if your CSV already uses 0/1)
    mapping = {
        "TUMOR": 1, "METASTASIS": 1, "POS": 1, "POSITIVE": 1, "1": 1,
        "NORMAL": 0, "NEG": 0, "NEGATIVE": 0, "0": 0
    }
    mapped = st.map(mapping)
    if mapped.isna().any():
        raise ValueError("Label normalization failed for some rows. "
                         "Ensure labels are 0/1 or use consistent tokens (e.g., TUMOR/NORMAL).")
    return mapped.astype(int)


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    df = pd.read_csv(args.csv)
    ensure_cols(df, ["slide_id", "full_path", "label", "split"])

    # Normalize split values and label dtype
    split_norm = df["split"].astype(str).str.lower()
    train_df = df[split_norm == "train"].copy().reset_index(drop=True)
    test_df  = df[split_norm == "test"].copy().reset_index(drop=True)

    train_df["label"] = normalize_label_series(train_df["label"])
    test_df["label"]  = normalize_label_series(test_df["label"])

    # Columns to write out (matches your tcga-brca fold format we discussed)
    out_cols = ["slide_id", "full_path", "label"]

    # Basic sanity checks
    if train_df.empty or test_df.empty:
        raise ValueError("Train or Test subset is empty. Check the 'split' column values in the CSV.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test_size must be in (0,1)")

    # Ensure we have exactly 5 seeds
    if len(args.seeds) != 5:
        raise ValueError(f"Please provide exactly 5 seeds via --seeds (got {len(args.seeds)}).")

    print(f"Total rows: {len(df)} | Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Test  labels: {test_df['label'].value_counts().to_dict()}")
    print(f"Writing folds under: {args.out_root}\n")

    # Create 5 independent stratified splits of train into 90/10 train/val
    for i, seed in enumerate(args.seeds, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        (tr_idx, va_idx), = sss.split(train_df, train_df["label"])

        tr_split = train_df.iloc[tr_idx].copy().reset_index(drop=True)
        va_split = train_df.iloc[va_idx].copy().reset_index(drop=True)
        te_split = test_df.copy().reset_index(drop=True)

        # Keep only desired columns
        tr_split = tr_split[out_cols]
        va_split = va_split[out_cols]
        te_split = te_split[out_cols]

        fold_dir = os.path.join(args.out_root, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        for name, split_df in [("train.csv", tr_split), ("val.csv", va_split), ("test.csv", te_split)]:
            out_path = os.path.join(fold_dir, name)
            if os.path.exists(out_path) and not args.force:
                raise FileExistsError(
                    f"{out_path} already exists. Use --force to overwrite, or delete it and rerun."
                )
            split_df.to_csv(out_path, index=False)

        print(
            f"Fold {i} (seed {seed}) "
            f"| Train={len(tr_split)} (pos={int(tr_split['label'].sum())}) "
            f"| Val={len(va_split)} (pos={int(va_split['label'].sum())}) "
            f"| Test={len(te_split)} (pos={int(te_split['label'].sum())})"
        )

    print("\n✅ Done. Folds written successfully.")


if __name__ == "__main__":
    main()
