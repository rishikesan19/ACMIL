#!/usr/bin/env python3
"""
Evaluate results from all folds and compute mean ± std (as in M3amba paper)
"""
import json
import numpy as np
import sys
import os

def load_fold_results(fold_dir):
    # Load your model's test results here
    # This is a template - modify based on your result format
    results_file = os.path.join(fold_dir, 'test_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

# Collect results from all folds
all_results = {
    'accuracy': [],
    'auc': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for fold in range(1, 6):
    fold_dir = f'results/fold_{fold}'
    results = load_fold_results(fold_dir)
    if results:
        for metric in all_results:
            if metric in results:
                all_results[metric].append(results[metric])

# Compute mean ± std as in paper
print("5-Fold Cross-Validation Results (M3amba paper format):")
print("="*50)
for metric, values in all_results.items():
    if values:
        mean = np.mean(values) * 100  # Convert to percentage
        std = np.std(values) * 100
        print(f"{metric.capitalize():10s}: {mean:.1f}% ± {std:.1f}%")

# Paper reports: "mean ± standard deviation across the five test sets"
