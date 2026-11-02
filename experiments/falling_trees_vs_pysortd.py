"""
Compare Falling Trees (C++ backend) vs pysortd.
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
from falling_trees import falling_rashomon_cpp as gravitree
from falling_trees.check_cpp_rset_falling import load_cpp_rset
from utils import (
    expected_decision_sparsity_falling_tree,
    compute_tree_test_loss_threshold,
)

# Import pysortd
sys.path.insert(0, "/usr/project/xtmp/vb97/FRL_Rashomon_Set/frl_rashomon_set/lib/python3.10/site-packages")
from pysortd import SORTDClassifier


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


def expected_decision_sparsity_pysortd(node, dataset: np.ndarray):
    """
    Compute the expected decision sparsity of a pysortd tree node given a dataset.
    Similar to expected_decision_sparsity_falling_tree but for pysortd tree structure.
    """
    if node.is_leaf_node():
        return 0.0
    
    current_feature = node.feature
    left_dataset = dataset[dataset[:, current_feature] == 0]
    right_dataset = dataset[dataset[:, current_feature] == 1]
    eps = 1e-6
    return (len(left_dataset) * (1 + expected_decision_sparsity_pysortd(node.left_child, left_dataset)) + 
            len(right_dataset) * (1 + expected_decision_sparsity_pysortd(node.right_child, right_dataset))) / (len(dataset) + eps)


def _safe_mean(values):
    return float(np.mean(values)) if values is not None and len(values) > 0 else 0.0


def _pysortd_metrics(
    model,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    max_models: int | None,
):
    """
    Compute metrics for pysortd model's Rashomon set.
    Similar to _treefarms_metrics but for pysortd.
    """
    total_models = model.rashomon_set_size
    if max_models is not None:
        total_models = min(total_models, max_models)

    X_test_arr = X_test_df.astype(bool).values
    y_test_arr = y_test
    pos_mask = y_test_arr == 1

    sparsity_all = []
    sparsity_pos = []
    losses = []
    losses_pos = []
    
    for idx in range(total_models):
        try:
            tree = model.get_tree_n(idx)
            if tree is None:
                break
            
            sparsity_all.append(expected_decision_sparsity_pysortd(tree, X_test_arr))
            if np.any(pos_mask):
                sparsity_pos.append(
                    expected_decision_sparsity_pysortd(tree, X_test_arr[pos_mask])
                )
            
            # Get predictions for this tree
            preds = model.predict(X_test_df, tree=tree)
            losses.append(float(np.mean(preds != y_test_arr)))
            if np.any(pos_mask):
                losses_pos.append(
                    float(np.mean(preds[pos_mask] != y_test_arr[pos_mask]))
                )
        except Exception as e:
            print(f"Warning: Error processing tree {idx}: {e}", flush=True)
            break

    return {
        "rset_size": model.rashomon_set_size,
        "sparsity_mean": _safe_mean(sparsity_all),
        "sparsity_pos_mean": _safe_mean(sparsity_pos),
        "loss_mean": _safe_mean(losses),
        "loss_pos_mean": _safe_mean(losses_pos),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Falling Trees (C++ backend) vs pysortd"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--label-column", type=str, default=None, help="Label column name")
    parser.add_argument("--num-estimators", type=int, default=50, help="GBDT estimators for binarization")
    parser.add_argument("--depth", type=int, default=5, help="Max depth of tree")
    parser.add_argument("--lam", type=float, default=0.02, help="Regularization parameter")
    parser.add_argument("--eps", type=float, default=0.01, help="Rashomon budget epsilon")
    parser.add_argument("--min-support", type=float, default=0.02, help="Minimum support for splits")
    parser.add_argument("--use-heap", action="store_true", default=True, help="Use heap in R-set search")
    parser.add_argument("--max-cache-size", type=int, default=10**7, help="Max subproblem cache size")
    parser.add_argument("--branching-costs", type=str, default=None, help="Single branching cost or comma-separated list")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of train/test splits")
    parser.add_argument("--max-pysortd-models", type=int, default=None, help="Limit pysortd models evaluated")
    parser.add_argument("--output-dir", type=str, default="falling_trees_vs_pysortd_results", help="Output directory")

    args = parser.parse_args()

    branching_costs = _parse_branching_costs(args.branching_costs)

    df = pd.read_csv(args.dataset)
    label_col = df.columns[-1] if args.label_column is None else args.label_column
    dataset_name = Path(args.dataset).stem

    print("Running experiments...", flush=True)
    print("pysortd will be imported when needed.", flush=True)
    
    for bc in branching_costs:
        # Create directory with format: falling_trees_vs_pysortd_results_{lam}_{eps}_mu
        # where mu is the branching cost (bc)
        output_dir = Path(f"falling_trees_vs_pysortd_results_{args.lam}_{args.eps}_{bc}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        detailed_rows = []
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

            # Run Falling Trees
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
                dump_path = tmp_file.name

            start_time = time.time()
            result = gravitree.run_from_xy(
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

            # Run pysortd
            # Map parameters: lam -> cost_complexity, eps -> rashomon_multiplier
            print("Training pysortd model...")
            pysortd_model = SORTDClassifier(
                "cost-complex-accuracy",
                max_depth=args.depth+1,
                verbose=False,
                cost_complexity=args.lam,  # Map lam to cost_complexity
                use_rashomon_multiplier=True,
                rashomon_multiplier=args.eps,  # Map eps to rashomon_multiplier
                max_num_trees=1000 if args.max_pysortd_models is None else args.max_pysortd_models,
            )
            start_time = time.time()
            pysortd_model.fit(X_train_df.astype(bool), y_train)
            pysortd_time = time.time() - start_time
            print(f"pysortd time: {pysortd_time}")
            print(f"pysortd R-set size: {pysortd_model.rashomon_set_size}")
            
            ps_metrics = _pysortd_metrics(
                pysortd_model,
                X_test_df,
                y_test,
                args.max_pysortd_models,
            )
            print(f"pysortd metrics: {ps_metrics}")

            detailed_rows.append(
                {
                    "dataset": dataset_name,
                    "split_idx": split_idx,
                    "branching_cost": bc,
                    "falling_trees_time": ft_time,
                    "pysortd_time": pysortd_time,
                    "falling_trees_rset_size": len(ft_rset),
                    "pysortd_rset_size": ps_metrics["rset_size"],
                    "falling_trees_sparsity_mean": _safe_mean(ft_sparsity),
                    "pysortd_sparsity_mean": ps_metrics["sparsity_mean"],
                    "falling_trees_sparsity_pos_mean": _safe_mean(ft_sparsity_pos),
                    "pysortd_sparsity_pos_mean": ps_metrics["sparsity_pos_mean"],
                    "falling_trees_loss_mean": _safe_mean(ft_loss),
                    "falling_trees_loss_pos_mean": _safe_mean(ft_loss_pos),
                    "pysortd_loss_mean": ps_metrics["loss_mean"],
                    "pysortd_loss_pos_mean": ps_metrics["loss_pos_mean"],
                }
            )

        # Save results for this branching cost
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df.to_csv(output_dir / f"{dataset_name}_full_detailed_results.csv", index=False)
        print(f"\nResults saved to {output_dir} for branching cost {bc}")


if __name__ == "__main__":
    main()

