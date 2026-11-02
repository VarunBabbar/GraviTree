"""Evaluate the stability of the falling tree Rashomon set across different random train/test splits.
This experiment assesses how much, on average, the Rashomon set changes when we vary the random seed for train/test splits.
The closeness of one tree to another is measured using the Hamming distance between their predictions on the test set. 
The stability is then quantified as the average Hamming distance between all pairs of tree in the two Rashomon sets obtained from different random seeds.
"""

import os
import sys
import time
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path so we can import falling_trees and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from falling_trees.binarize_dataset import binarize_dataset
from falling_trees import falling_rashomon_cpp as gravitree
from falling_trees.gravitree import GraviTree


def load_dataset(data_path: str, label_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset from a CSV file."""
    data = pd.read_csv(data_path)
    if label_column is None:
        label_column = data.columns[-1]  # Assume the last column is the label if not specified
    X = data.drop(columns=[label_column])
    y = data[label_column]
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.", flush=True)
    print(f"Label column: {label_column}", flush=True)
    return X, y


def binarize_X_y(X_train, y_train, X_test, y_test, num_estimators=50):
    """Binarize the features and labels using the binarization method from falling_trees."""

    df_train = X_train.copy()
    df_train['label'] = y_train
    df_binary, thresholds, header, _ = binarize_dataset(df_train, num_estimators=num_estimators)
    X_train_binary = df_binary.drop(columns=['label'])
    y_train_binary = df_binary['label']

    df_test = X_test.copy()
    df_test['label'] = y_test
    df_test_binary = binarize_dataset(df_test, num_estimators=num_estimators, thresholds=thresholds, header=header)
    X_test_binary = df_test_binary.drop(columns=['label'])
    y_test_binary = df_test_binary['label']
    return X_train_binary, y_train_binary, X_test_binary, y_test_binary, thresholds, header


def train_test_split_and_binarize(X, y, test_size=0.2, random_state=None, num_estimators=50):
    """Split the data into train and test sets and binarize them."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_bin, y_train_bin, X_test_bin, y_test_bin, thresholds, header \
        = binarize_X_y(X_train,
                       y_train,
                       X_test,
                       y_test,
                       num_estimators=num_estimators)
    return X_train_bin, y_train_bin, X_test_bin, y_test_bin, thresholds, header


def evaluate_rashomon_set_stability(X, y, num_seeds=5):
    """Evaluate the stability of the Rashomon set across different random seeds."""
    mean_hamming_distances = []

    for seed in range(0, num_seeds * 3, 3):
        print(f"Evaluating stability for seed {seed//3}...")

        seed_hamming_distances = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        X_train1_bin, y_train1_bin, _, _, thresholds1, header1 \
            = train_test_split_and_binarize(X_train,
                                            y_train,
                                            random_state=seed + 1,
                                            num_estimators=50)
        X_train2_bin, y_train2_bin, _, _, thresholds2, header2 \
            = train_test_split_and_binarize(X_train,
                                            y_train,
                                            random_state=seed + 2,
                                            num_estimators=50)

        print(f"Training GraviTree for seed {seed+1}...", flush=True)
        gravitree1 = GraviTree(eps=0.02, depth=5, branching_cost=0.01, min_support=0.04)
        gravitree1.fit(X_train1_bin, y_train1_bin)
        print(f"GraviTree for seed {seed + 1} trained with {len(gravitree1.ft_rset)} trees in the Rashomon set.",
              flush=True)

        print(f"Training GraviTree for seed {seed+2}...", flush=True)
        gravitree2 = GraviTree(eps=0.02, depth=5, branching_cost=0.01, min_support=0.04)
        gravitree2.fit(X_train2_bin, y_train2_bin)
        print(f"GraviTree for seed {seed + 2} trained with {len(gravitree2.ft_rset)} trees in the Rashomon set.",
              flush=True)

        test_df = pd.concat([X_test, y_test], axis=1)
        test1_bin = binarize_dataset(test_df, num_estimators=50, thresholds=thresholds1, header=header1)
        test2_bin = binarize_dataset(test_df, num_estimators=50, thresholds=thresholds2, header=header2)
        test1_X = test1_bin.iloc[:, :-1]
        test2_X = test2_bin.iloc[:, :-1]

        print(f"Making predictions for seed {seed + 1}...", flush=True)
        predictions1 = [gravitree1.predict(test1_X, tree_idx=i) for i in range(len(gravitree1.ft_rset))]
        print(f"Making predictions for seed {seed + 2}...", flush=True)
        predictions2 = [gravitree2.predict(test2_X, tree_idx=i) for i in range(len(gravitree2.ft_rset))]

        print(f"Computing Hamming distances for seed {seed//3}...", flush=True)
        # for each model in the Rashomon set of seed 1, find the model in the Rashomon set of seed 2
        # with the smallest Hamming distance
        for i, pred1 in enumerate(predictions1):
            min_hamming_distance = float("inf")
            for j, pred2 in enumerate(predictions2):
                hamming_distance = np.mean(pred1 != pred2)
                if hamming_distance < min_hamming_distance:
                    min_hamming_distance = hamming_distance
            seed_hamming_distances.append(min_hamming_distance)

        mean_hamming_distance = np.mean(seed_hamming_distances)
        mean_hamming_distances.append(mean_hamming_distance)
        print(
            f"Seed {seed//3}: Mean min Hamming distance between trees in the Rashomon set: {mean_hamming_distance:.4f}",
            flush=True)

    # Compute the average Hamming distance as a measure of stability
    return mean_hamming_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the stability of the falling tree Rashomon set across different random seeds.")
    parser.add_argument("--data_path",
                        type=str,
                        default="data/falling_trees.csv",
                        help="Path to the falling trees dataset CSV file.")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds to evaluate.")
    parser.add_argument("--label_column", type=str, default=None, help="Name of the label column in the dataset.")
    args = parser.parse_args()
    dataset_name = Path(args.data_path).stem

    print(f"Loading {args.data_path}...", flush=True)

    # Load the dataset
    X, y = load_dataset(args.data_path, label_column=args.label_column)

    # Evaluate the stability of the Rashomon set
    mean_hamming_distances = evaluate_rashomon_set_stability(X, y, num_seeds=args.num_seeds)

    results_df = pd.DataFrame({"seed": list(range(args.num_seeds)), "mean_hamming_distance": mean_hamming_distances})
    results_path = f"results/rashomon_set_stability_{dataset_name}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Stability results saved to {results_path}", flush=True)
