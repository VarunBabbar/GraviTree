"""
Ablation study for GraviTree algorithm.
Tests various aspects: falling constraint, quantized caching, and recursion order.

Given a branching cost mu, tests:
1. Runtime, avg loss, and rset size with vs without falling constraint
2. Quantized caching levels (1, 0.1, 0.01, 0.001) with falling constraint on
3. Recursing on lower probability vs higher probability leaf
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
from falling_trees.check_cpp_rset_falling import load_cpp_rset
from falling_trees.frl_rashomon_set_alg_falling_constraint_v2 import (
    normalized_colless_index,
)
from utils import (
    expected_decision_sparsity_falling_tree,
    compute_tree_test_loss_threshold,
)


def rebalance_if_imbalanced(df, label_col, random_state=None):
    """Minority oversampling if class balance exceeds 70/30."""
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        return df
    majority_count = counts.max()
    minority_count = counts.min()
    total = len(df)
    if total == 0 or minority_count == 0:
        return df
    majority_frac = majority_count / total
    minority_frac = minority_count / total
    if majority_frac <= 0.7 and minority_frac >= 0.3:
        return df
    minority_label = counts.idxmin()
    minority_df = df[df[label_col] == minority_label]
    add_n = majority_count - minority_count
    if add_n <= 0:
        return df
    sampled = minority_df.sample(n=add_n, replace=True, random_state=random_state)
    return pd.concat([df, sampled], ignore_index=True)


def _safe_mean(values):
    """Compute mean safely, handling empty arrays."""
    return float(np.mean(values)) if values is not None and len(values) > 0 else 0.0


def run_gravitree_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    branching_cost: float,
    depth: int,
    lam: float,
    eps: float,
    min_support: float,
    enable_falling_constraint: bool,
    max_cache_size: int,
    quantized_cache_level: Optional[float] = None,
    recurse_on_lower_prob_first: bool = False,
    budget_override: Optional[float] = None,
) -> Dict:
    """
    Run GraviTree experiment with specified parameters.
    
    Parameters:
    -----------
    quantized_cache_level : float, optional
        Quantization level for caching. If None, no quantization.
        Values like 1, 0.1, 0.01, 0.001 round cache keys to that precision.
    """
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        dump_path = tmp_file.name

    try:
        # Build kwargs with only supported parameters.
        if quantized_cache_level is not None:
            cache_bucket_size = quantized_cache_level
        else:
            cache_bucket_size = 1e-3
        
        start_time = time.time()
        print("Running gravitree with the following parameters:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"lam: {lam}")
        print(f"eps: {eps}")
        print(f"depth: {depth}")
        print(f"enable_falling_constraint: {enable_falling_constraint}")
        print(f"use_heap: True")
        print(f"branching_cost: {branching_cost}")
        print(f"min_support: {min_support}")
        print(f"use_current_leaf_prob: True")
        print(f"budget_override: {budget_override}")
        print(f"dump_rset_jsonl: {dump_path}")
        print(f"max_cache_size: {max_cache_size}")
        print(f"cache_bucket_size: {cache_bucket_size}")
        print(f"recurse_on_lower_prob_first: {recurse_on_lower_prob_first}")
        result = gravitree.run_from_xy(
            X_train,
            y_train,
            lam=lam,
            eps=eps,
            depth=depth,
            enable_falling_constraint=enable_falling_constraint,
            use_heap=True,
            branching_cost=branching_cost,
            min_support=min_support,
            use_current_leaf_prob=True,
            budget_override=budget_override,
            dump_rset_jsonl=dump_path,
            max_cache_size=max_cache_size,
            cache_bucket_size=cache_bucket_size,
            recurse_on_lower_prob_first = recurse_on_lower_prob_first,
        )
        runtime = time.time() - start_time
        
        ft_rset = load_cpp_rset(dump_path)
        rset_size = len(ft_rset)
        
        # Compute metrics on test set
        X_test_bool = X_test.astype(bool)
        pos_mask = y_test == 1
        
        # Compute losses for all models in R-set
        losses = [
            compute_tree_test_loss_threshold(model[0], X_test, y_test, threshold=0.5)
            for model in ft_rset
        ]
        
        avg_loss = _safe_mean(losses)
        
        # Compute sparsity (optional, for completeness)
        sparsity = [
            expected_decision_sparsity_falling_tree(model[0], X_test_bool)
            for model in ft_rset
        ]
        avg_sparsity = _safe_mean(sparsity)

        normalized_colless_vals = [
            normalized_colless_index(model[0]) for model in ft_rset
        ]
        avg_normalized_colless = _safe_mean(normalized_colless_vals)

        return {
            "runtime": runtime,
            "rset_size": rset_size,
            "avg_loss": avg_loss,
            "avg_sparsity": avg_sparsity,
            "avg_normalized_colless": avg_normalized_colless,
            "best_loss": float(result.get("best_loss", 0.0)),
        }
        
    finally:
        if os.path.exists(dump_path):
            os.remove(dump_path)


def run_ablation_study(
    dataset_path: str,
    branching_cost: float,
    depth: int = 5,
    lam: float = 0.02,
    eps: float = 0.01,
    min_support: float = 0.05,
    num_trials: int = 5,
    max_cache_size: int = 10**7,
    num_estimators: int = 100,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run ablation study for GraviTree algorithm.
    
    Tests:
    1. With vs without falling constraint
    2. Quantized caching levels (1, 0.1, 0.01, 0.001) with falling constraint on
    3. Recursing on lower probability vs higher probability leaf
    """
    df = pd.read_csv(dataset_path)
    label_col = df.columns[-1]
    dataset_name = Path(dataset_path).stem
    
    if output_dir is None:
        output_dir = Path(f"ablation_study_results_mu_{branching_cost}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"Ablation Study: Dataset={dataset_name}, Branching Cost (mu)={branching_cost}")
    print(f"{'='*80}\n")
    
    for trial_idx in range(num_trials):
        print(f"\nTrial {trial_idx + 1}/{num_trials}")
        print("-" * 80)
        
        # Create train/test split
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=trial_idx,
            stratify=df[label_col],
        )
        
        train_df = rebalance_if_imbalanced(train_df, label_col, random_state=trial_idx)
        
        # Binarize dataset
        train_bin, thresholds, header, _ = binarize_dataset(
            train_df, num_estimators=num_estimators
        )
        test_bin = binarize_dataset(
            test_df,
            num_estimators=num_estimators,
            thresholds=thresholds,
            header=header,
        )
        
        X_train_df = train_bin.drop(columns=[label_col])
        y_train = train_bin[label_col].astype(int).values
        X_test_df = test_bin.drop(columns=[label_col])
        y_test = test_bin[label_col].astype(int).values
        
        X_train = X_train_df.values
        X_test = X_test_df.values
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # ====================================================================
        # Test 1: With vs Without Falling Constraint
        # ====================================================================
        print("\n1. Testing with vs without falling constraint...")
        
        # With falling constraint, higher prob leaf
        print("  - Falling constraint ON, higher prob leaf...")
        result_falling_on = run_gravitree_experiment(
            X_train, y_train, X_test, y_test,
            branching_cost=branching_cost,
            depth=depth,
            lam=lam,
            eps=eps,
            min_support=min_support,
            enable_falling_constraint=True,
            max_cache_size=max_cache_size,
            recurse_on_lower_prob_first = False,
        )
        base_best_loss = float(result_falling_on.get("best_loss", 0.0))
        base_budget = base_best_loss * (1.0 + eps)
        results.append({
            "trial": trial_idx,
            "dataset": dataset_name,
            "branching_cost": branching_cost,
            "config": "falling_constraint_on_higher_prob",
            "enable_falling_constraint": True,
            "use_current_leaf_prob": True,
            "quantized_cache_level": None,
            **result_falling_on,
        })
        
        # # Without falling constraint, higher prob leaf
        # print("  - Falling constraint OFF, higher prob leaf...")
        # result_falling_off = run_gravitree_experiment(
        #     X_train, y_train, X_test, y_test,
        #     branching_cost=branching_cost,
        #     depth=depth,
        #     lam=lam,
        #     eps=eps,
        #     min_support=min_support,
        #     enable_falling_constraint=False,
        #     max_cache_size=max_cache_size,
        #     recurse_on_lower_prob_first = False,
        #     budget_override=base_budget,
        # )
        # results.append({
        #     "trial": trial_idx,
        #     "dataset": dataset_name,
        #     "branching_cost": branching_cost,
        #     "config": "falling_constraint_on_vs_off",
        #     "enable_falling_constraint": True,
        #     "use_current_leaf_prob": True,
        #     "quantized_cache_level": None,
        #     **result_falling_off,
        # })

        print("\n2. Testing quantized caching levels (falling constraint ON)...")
        quantized_levels = [1.0, 0.1, 0.01, 0.001]
        
        for q_level in quantized_levels:
            print(f"  - Quantized cache level: {q_level}...")
            result_qcache = run_gravitree_experiment(
                X_train, y_train, X_test, y_test,
                branching_cost=branching_cost,
                depth=depth,
                lam=lam,
                eps=eps,
                min_support=min_support,
                enable_falling_constraint=True,
                max_cache_size=max_cache_size,
                quantized_cache_level=q_level,
                recurse_on_lower_prob_first = False,
                budget_override=base_budget,
            )
            results.append({
                "trial": trial_idx,
                "dataset": dataset_name,
                "branching_cost": branching_cost,
                "config": f"quantized_cache_{q_level}",
                "enable_falling_constraint": True,
                "use_current_leaf_prob": True,
                "quantized_cache_level": q_level,
                **result_qcache,
            })
        
        # ====================================================================
        # Test 3: Recursion Order (lower vs higher probability leaf)
        # ====================================================================
        # print("\n3. Testing recursion order (lower vs higher prob leaf)...")
        
        # Higher probability leaf (already tested above, but include for completeness)
        # Lower probability leaf
        # print("  - Lower prob leaf (falling constraint ON)...")
        # result_lower_prob = run_gravitree_experiment(
        #     X_train, y_train, X_test, y_test,
        #     branching_cost=branching_cost,
        #     depth=depth,
        #     lam=lam,
        #     eps=eps,
        #     min_support=min_support,
        #     enable_falling_constraint=True,
        #     max_cache_size=max_cache_size,
        #     recurse_on_lower_prob_first = True,
        #     budget_override=base_budget,
        # )
        # results.append({
        #     "trial": trial_idx,
        #     "dataset": dataset_name,
        #     "branching_cost": branching_cost,
        #     "config": "falling_constraint_on_lower_prob",
        #     "enable_falling_constraint": True,
        #     "use_current_leaf_prob": False,
        #     "quantized_cache_level": None,
        #     **result_lower_prob,
        # })
        
        print(f"\nTrial {trial_idx + 1} completed.")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = output_dir / f"{dataset_name}_ablation_detailed_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Compute summary statistics
    summary_rows = []
    for config in results_df["config"].unique():
        config_data = results_df[results_df["config"] == config]
        summary_rows.append({
            "config": config,
            "dataset": dataset_name,
            "branching_cost": branching_cost,
            "mean_runtime": config_data["runtime"].mean(),
            "std_runtime": config_data["runtime"].std(),
            "mean_rset_size": config_data["rset_size"].mean(),
            "std_rset_size": config_data["rset_size"].std(),
            "mean_avg_loss": config_data["avg_loss"].mean(),
            "std_avg_loss": config_data["avg_loss"].std(),
            "mean_avg_sparsity": config_data["avg_sparsity"].mean(),
            "std_avg_sparsity": config_data["avg_sparsity"].std(),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / f"{dataset_name}_ablation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary results saved to: {summary_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for GraviTree algorithm"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )
    parser.add_argument(
        "--branching-cost",
        type=float,
        required=True,
        help="Branching cost mu to test"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Max depth of tree (default: 5)"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.02,
        help="Regularization parameter (default: 0.02)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Rashomon budget epsilon (default: 0.01)"
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Minimum support for splits (default: 0.05)"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of train/test splits (default: 5)"
    )
    parser.add_argument(
        "--max-cache-size",
        type=int,
        default=10**7,
        help="Max subproblem cache size (default: 10^7)"
    )
    parser.add_argument(
        "--num-estimators",
        type=int,
        default=100,
        help="GBDT estimators for binarization (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ablation_study_results_mu_{branching_cost})"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GraviTree Ablation Study")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Branching Cost (mu): {args.branching_cost}")
    print(f"Depth: {args.depth}")
    print(f"Lambda: {args.lam}")
    print(f"Epsilon: {args.eps}")
    print(f"Min Support: {args.min_support}")
    print(f"Number of Trials: {args.num_trials}")
    print("=" * 80)
    
    results_df = run_ablation_study(
        dataset_path=args.dataset,
        branching_cost=args.branching_cost,
        depth=args.depth,
        lam=args.lam,
        eps=args.eps,
        min_support=args.min_support,
        num_trials=args.num_trials,
        max_cache_size=args.max_cache_size,
        num_estimators=args.num_estimators,
        output_dir=args.output_dir,
    )
    
    print("\n" + "=" * 80)
    print("Ablation Study Completed!")
    print("=" * 80)
    
    # Print summary
    print("\nSummary by Configuration:")
    print("-" * 80)
    summary = results_df.groupby("config").agg({
        "runtime": ["mean", "std"],
        "rset_size": ["mean", "std"],
        "avg_loss": ["mean", "std"],
    }).round(4)
    print(summary)


if __name__ == "__main__":
    main()

