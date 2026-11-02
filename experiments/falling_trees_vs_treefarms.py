"""
Compare Falling Trees (C++ backend) vs TREEFARMS.
Tracks runtime, R-set size, decision sparsity, and loss across branching costs.
"""

import os
import sys
import time
import tempfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path so we can import falling_trees and resplit
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from falling_trees.binarize_dataset import binarize_dataset
import falling_trees.gravitree_cpp as gravitree_cpp
from falling_trees.check_cpp_rset_falling import load_cpp_rset
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


def _parse_branching_costs(costs_str: str):
    if costs_str is None:
        return [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    costs_str = costs_str.strip()
    if "," in costs_str:
        return [float(x.strip()) for x in costs_str.split(",")]
    return [float(costs_str)]


def _treefarms_path_length(tree, sample):
    node = tree.source
    depth = 0
    while "prediction" not in node:
        feat = node["feature"]
        depth += 1
        if sample[feat] == 1:
            node = node["true"]
        else:
            node = node["false"]
    return depth


def expected_decision_sparsity_treefarms(tree, X):
    if X.size == 0:
        return 0.0
    return float(np.mean([_treefarms_path_length(tree, row) for row in X]))


def _safe_mean(values):
    return float(np.mean(values)) if values is not None and len(values) > 0 else 0.0


def _treefarms_metrics(
    model,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    max_models: int | None,
):
    total_models = model.get_tree_count()
    if max_models is not None:
        total_models = min(total_models, max_models)

    X_test_arr = X_test_df.values
    y_test_arr = y_test
    pos_mask = y_test_arr == 1

    sparsity_all = []
    sparsity_pos = []
    losses = []
    losses_pos = []
    for idx in range(total_models):
        tree = model[idx]
        sparsity_all.append(expected_decision_sparsity_treefarms(tree, X_test_arr))
        if np.any(pos_mask):
            sparsity_pos.append(
                expected_decision_sparsity_treefarms(tree, X_test_arr[pos_mask])
            )
        preds = tree.predict(X_test_df)
        losses.append(float(np.mean(preds != y_test_arr)))
        if np.any(pos_mask):
            losses_pos.append(
                float(np.mean(preds[pos_mask] != y_test_arr[pos_mask]))
            )

    return {
        "rset_size": model.get_tree_count(),
        "sparsity_mean": _safe_mean(sparsity_all),
        "sparsity_pos_mean": _safe_mean(sparsity_pos),
        "loss_mean": _safe_mean(losses),
        "loss_pos_mean": _safe_mean(losses_pos),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Falling Trees (C++ backend) vs TREEFARMS"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--label-column", type=str, default=None, help="Label column name")
    parser.add_argument("--num-estimators", type=int, default=50, help="GBDT estimators for binarization")
    parser.add_argument("--depth", type=int, default=5, help="Max depth of tree")
    parser.add_argument("--lam", type=float, default=0.005, help="Regularization parameter")
    parser.add_argument("--eps", type=float, default=0.01, help="Rashomon budget epsilon")
    parser.add_argument("--min-support", type=float, default=0.02, help="Minimum support for splits")
    parser.add_argument("--use-heap", action="store_true", default=True, help="Use heap in R-set search")
    parser.add_argument("--max-cache-size", type=int, default=10**7, help="Max subproblem cache size")
    parser.add_argument("--branching-costs", type=str, default=None, help="Single branching cost or comma-separated list")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of train/test splits")
    parser.add_argument("--max-treefarms-models", type=int, default=None, help="Limit TREEFARMS models evaluated")
    parser.add_argument("--output-dir", type=str, default="falling_trees_vs_treefarms_results", help="Output directory")

    args = parser.parse_args()

    branching_costs = _parse_branching_costs(args.branching_costs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    label_col = df.columns[-1] if args.label_column is None else args.label_column
    dataset_name = Path(args.dataset).stem

    detailed_rows = []
    print("Running experiments...", flush=True)
    print("Loading TREEFARMS...", flush=True)
    from resplit import TREEFARMS
    print("Loaded TREEFARMS.", flush=True)
    
    for bc in branching_costs:
        for split_idx in range(args.num_trials):
            print(f"\n{'='*80}")
            print(f"Branching Cost {bc} | Split {split_idx + 1}/{args.num_trials}")
            print(f"{'='*80}")

            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=split_idx,
                stratify=df[label_col],
            )
            
            train_df = rebalance_if_imbalanced(train_df, label_col, random_state=split_idx)

            train_bin, thresholds, header, _ = binarize_dataset(
                train_df, num_estimators=args.num_estimators
            )
            print("Shape of input data:", train_bin.shape, flush=True)
            test_bin = binarize_dataset(
                test_df,
                num_estimators=args.num_estimators,
                thresholds=thresholds,
                header=header,
            )

            X_train_df = train_bin.drop(columns=[label_col])
            y_train = train_bin[label_col].astype(int)
            X_test_df = test_bin.drop(columns=[label_col])
            y_test = test_bin[label_col].astype(int).values

            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
                dump_path = tmp_file.name

            start_time = time.time()
            result = gravitree_cpp.run_from_xy(
                X_train_df.values,
                y_train.values,
                lam=args.lam,
                eps=args.eps,
                depth=args.depth,
                enable_falling_constraint=True,
                use_heap=args.use_heap,
                rule_list_mode=False,
                branching_cost=bc,
                min_support=args.min_support,
                use_current_leaf_prob=True,
                budget_override=None,
                dump_rset_jsonl=dump_path,
                max_cache_size=args.max_cache_size,
            )
            ft_time = time.time() - start_time
            print(f"GraviTree time: {ft_time}")
            ft_rset = load_cpp_rset(dump_path)
            print(f"GraviTree R-set size: {len(ft_rset)}")
            os.remove(dump_path)

            X_test_bool = X_test_df.astype(bool).values
            pos_mask = y_test == 1
            ft_sparsity = [
                expected_decision_sparsity_falling_tree(model[0], X_test_bool)
                for model in ft_rset
            ]
            ft_sparsity_pos = [
                expected_decision_sparsity_falling_tree(
                    model[0], X_test_bool[pos_mask]
                )
                for model in ft_rset
            ] if np.any(pos_mask) else []
            ft_loss = [
                compute_tree_test_loss_threshold(model[0], X_test_df.values, y_test, threshold=0.5)
                for model in ft_rset
            ]
            ft_loss_pos = [
                compute_tree_test_loss_threshold(
                    model[0], X_test_df.values[pos_mask], y_test[pos_mask], threshold=0.5
                )
                for model in ft_rset
            ] if np.any(pos_mask) else []

            treefarms_config = {
                "regularization": args.lam,
                "depth_budget": args.depth+1, # correcting for off by one in TREEFARMS
                "rashomon_bound_multiplier": args.eps,
                "time_limit": 100,
            }
            treefarms_model = TREEFARMS(treefarms_config)
            print("Training TREEFARMS model...")
            start_time = time.time()
            treefarms_model.fit(X_train_df.astype(bool), y_train.astype(bool))
            treefarms_time = time.time() - start_time
            print(f"TREEFARMS time: {treefarms_time}")
            tf_metrics = _treefarms_metrics(
                treefarms_model,
                X_test_df,
                y_test,
                args.max_treefarms_models,
            )
            print(f"TREEFARMS metrics: {tf_metrics}")

            detailed_rows.append(
                {
                    "dataset": dataset_name,
                    "split_idx": split_idx,
                    "branching_cost": bc,
                    "falling_trees_time": ft_time,
                    "treefarms_time": treefarms_time,
                    "falling_trees_rset_size": len(ft_rset),
                    "treefarms_rset_size": tf_metrics["rset_size"],
                    "falling_trees_sparsity_mean": _safe_mean(ft_sparsity),
                    "treefarms_sparsity_mean": tf_metrics["sparsity_mean"],
                    "falling_trees_sparsity_pos_mean": _safe_mean(ft_sparsity_pos),
                    "treefarms_sparsity_pos_mean": tf_metrics["sparsity_pos_mean"],
                    "falling_trees_loss_mean": _safe_mean(ft_loss),
                    "falling_trees_loss_pos_mean": _safe_mean(ft_loss_pos),
                    "treefarms_loss_mean": tf_metrics["loss_mean"],
                    "treefarms_loss_pos_mean": tf_metrics["loss_pos_mean"],
                }
            )

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(output_dir / f"{dataset_name}_full_detailed_results.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

