#!/usr/bin/env python3
"""
create_randomized_splits.py

Create 5 independent, randomized splits of the data with a 65:10:25 ratio.
This matches the methodology of running 5 separate trials, not a strict
cross-validation partition.

python create_randomized_splits_tcga_brca.py \
    --input ./dataset_csv/brca_idc_ilc_pure.csv \
    --output-dir dataset_csv/random_splits_tcga_brca \
    --seed 42
"""

import os
import csv
import argparse
import numpy as np
from collections import defaultdict
import random

def read_csv(filepath):
    """Read CSV and return rows"""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def create_randomized_splits(input_csv, output_dir, n_splits=5, seed=42):
    """
    Creates multiple independent, randomized splits of the entire dataset.
    
    For each of the `n_splits`, this function will randomly shuffle and partition
    the complete dataset into training (65%), validation (10%), and test (25%) sets.
    This is different from a k-fold cross-validation where test sets are mutually exclusive.
    
    Args:
        input_csv: Path to the main CSV file with all data.
        output_dir: Directory to save split CSVs.
        n_splits: The number of independent splits to create.
        seed: Random seed for reproducibility.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Read data
    data = read_csv(input_csv)
    print(f"Total samples: {len(data)}")
    print(f"Creating {n_splits} independent randomized splits with a 65:10:25 ratio.\n")
    
    # Group by patient to ensure patient-level splits
    patient_data = defaultdict(list)
    for row in data:
        patient_data[row['patient_id']].append(row)
    
    # Get unique patients and their labels
    patients = []
    patient_labels = []
    patient_rows = []
    
    for patient_id, rows in patient_data.items():
        patients.append(patient_id)
        patient_labels.append(rows[0]['label'])
        patient_rows.append(rows)
    
    patients = np.array(patients)
    patient_labels = np.array(patient_labels)
    
    unique_labels = np.unique(patient_labels)
    
    print("Patient-level class distribution:")
    for label in unique_labels:
        count = np.sum(patient_labels == label)
        print(f"  {label}: {count} patients")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group patient indices by label for stratification
    label_indices = {label: np.where(patient_labels == label)[0] for label in unique_labels}
    
    fold_stats = []
    
    # Main loop to create each independent split
    for split_idx in range(n_splits):
        print(f"\n{'='*60}")
        print(f"Creating Split {split_idx+1}/{n_splits}")
        print(f"{'='*60}")
        
        train_patient_indices, val_patient_indices, test_patient_indices = [], [], []
        
        # Perform a new stratified shuffle and split for each run
        for label in unique_labels:
            indices_for_label = label_indices[label].copy()
            np.random.shuffle(indices_for_label) # Re-shuffle for each split
            
            n_label = len(indices_for_label)
            
            # Calculate split points
            n_test = int(round(n_label * 0.25))
            n_val = int(round(n_label * 0.10))
            
            # Slice the shuffled array to get patient indices for each set
            test_indices_for_label = indices_for_label[:n_test]
            val_indices_for_label = indices_for_label[n_test : n_test + n_val]
            train_indices_for_label = indices_for_label[n_test + n_val :]
            
            # Append to the main index lists for this split
            test_patient_indices.extend(test_indices_for_label)
            val_patient_indices.extend(val_indices_for_label)
            train_patient_indices.extend(train_indices_for_label)
            
        # Create split directory
        split_dir = os.path.join(output_dir, f'split_{split_idx+1}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Collect all slides for each split
        train_slides = [slide for idx in train_patient_indices for slide in patient_rows[idx]]
        val_slides = [slide for idx in val_patient_indices for slide in patient_rows[idx]]
        test_slides = [slide for idx in test_patient_indices for slide in patient_rows[idx]]
        
        # Get the fieldnames from the original CSV
        fieldnames = data[0].keys()

        # Write CSVs for the current split
        for split_name, split_data in [('train', train_slides), ('val', val_slides), ('test', test_slides)]:
            file_path = os.path.join(split_dir, f'{split_name}.csv')
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(split_data)

        # Calculate statistics
        stats = {
            'split': split_idx + 1,
            'train_total': len(train_slides),
            'val_total': len(val_slides),
            'test_total': len(test_slides),
            'train_ratio': len(train_slides) / len(data) * 100,
            'val_ratio': len(val_slides) / len(data) * 100,
            'test_ratio': len(test_slides) / len(data) * 100,
        }
        fold_stats.append(stats)
        
        # Print split statistics
        print(f"\nSplit {split_idx+1} Statistics:")
        print(f"  Train: {stats['train_total']:4d} slides ({stats['train_ratio']:.1f}%)")
        print(f"  Val:   {stats['val_total']:4d} slides ({stats['val_ratio']:.1f}%)")
        print(f"  Test:  {stats['test_total']:4d} slides ({stats['test_ratio']:.1f}%)")

    # Write summary statistics
    summary_file = os.path.join(output_dir, 'splits_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['split', 'train_total', 'val_total', 'test_total',
                     'train_ratio', 'val_ratio', 'test_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_stats)
    
    print(f"\n{'='*60}")
    print(f"✓ Created {n_splits} independent splits in: {output_dir}")
    print(f"✓ Summary saved to: {summary_file}")
    print(f"✓ Each split follows the 65:10:25 ratio.")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Create 5 independent randomized splits of a dataset with a 65:10:25 ratio.")
    parser.add_argument('--input', required=True,
                        help='Input CSV file with all data')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for fold splits')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    create_randomized_splits(
        input_csv=args.input,
        output_dir=args.output_dir,
        n_splits=5,
        seed=args.seed
    )

if __name__ == '__main__':
    main()