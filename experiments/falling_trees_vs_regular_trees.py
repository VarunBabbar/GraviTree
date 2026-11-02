"""
Compare Falling Trees Rashomon sets with falling constraint on vs off.
Sweeps branching costs and compares expected decision sparsity.
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

# Add parent directory to path so we can import falling_trees
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from falling_trees.binarize_dataset import binarize_dataset
import falling_trees.frl_rashomon_set_alg as frl_rashomon_set_alg
from falling_trees.frl_rashomon_set_alg import (
    OptFallingTree,
    OptFallingRset,
    Leaf,
    tree_obj,
    _subproblem_optimal_objectives,
)
from utils import expected_decision_sparsity_falling_tree

def _is_binary_series(series: pd.Series) -> bool:
    unique_vals = pd.unique(series.dropna())
    return np.isin(unique_vals, [0, 1, False, True]).all()

def load_and_binarize_dataset(dataset_path: str, label_column: str, num_estimators: int):
    df = pd.read_csv(dataset_path)
    if label_column is None:
        label_column = df.columns[-1]
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")

    feature_cols = [c for c in df.columns if c != label_column]
    df = df[feature_cols + [label_column]]

    is_binary = all(_is_binary_series(df[col]) for col in feature_cols)
    if not is_binary:
        print("Binarizing dataset...")
        df, thresholds, header, threshold_guess_time = binarize_dataset(
            df, num_estimators=num_estimators
        )

    X = df.iloc[:, :-1].astype(int).values
    y = df.iloc[:, -1].astype(int).values
    return X, y, label_column


def run_rashomon_set(
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
    use_heap: bool,
    rule_list_mode: bool,
    max_cache_size: int,
    budget_override: float = None,
):
    frl_rashomon_set_alg.MAX_CACHE_SIZE = max_cache_size
    _subproblem_optimal_objectives.clear()

    n_train = X_train.shape[0]
    row_idx_train = np.arange(n_train)
    features = list(range(X_train.shape[1]))

    kwargs = {
        "branching_cost": branching_cost,
        "min_support": min_support,
    }
    start_time = time.time()
    best_loss, best_tree, pmax, depth_of_pmax = OptFallingTree(
        X_train,
        y_train,
        row_idx_train,
        depth,
        lam,
        features,
        n=n_train,
        enable_falling_constraint=enable_falling_constraint,
        rule_list_mode=rule_list_mode,
        **kwargs,
    )
    opt_tree_time = time.time() - start_time

    if budget_override is not None:
        budget = budget_override
    else:
        budget = best_loss * (1 + eps)

    start_time = time.time()
    R = OptFallingRset(
        X_train,
        y_train,
        row_idx_train,
        depth,
        lam,
        B=budget,
        features=features,
        n=n_train,
        enable_falling_constraint=enable_falling_constraint,
        use_heap=use_heap,
        rule_list_mode=rule_list_mode,
        **kwargs,
    )
    rset_time = time.time() - start_time
    print("Number of models in Rset: {}".format(len(R)))
    X_test_bool = X_test.astype(bool)
    terms = [expected_decision_sparsity_falling_tree(model[0], X_test_bool) for model in R]
    terms_pos = [
        expected_decision_sparsity_by_class_falling_tree(
            model[0], X_test_bool, y_test, 1
        )
        for model in R
    ]
    terms_neg = [
        expected_decision_sparsity_by_class_falling_tree(
            model[0], X_test_bool, y_test, 0
        )
        for model in R
    ]
    loss_train = [tree_obj(model[0]) for model in R]
    loss_pos = [
        compute_tree_loss_by_class(model[0], X_test, y_test, 1) for model in R
    ]
    loss_neg = [
        compute_tree_loss_by_class(model[0], X_test, y_test, 0) for model in R
    ]

    return {
        "best_loss": best_loss,
        "budget": budget,
        "rset_size": len(R),
        "terms": terms,
        "terms_pos": terms_pos,
        "terms_neg": terms_neg,
        "loss_train": loss_train,
        "loss_pos": loss_pos,
        "loss_neg": loss_neg,
        "opt_tree_time": opt_tree_time,
        "rset_time": rset_time,
        "total_time": opt_tree_time + rset_time,
    }


def run_comparison_for_branching_cost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    branching_cost: float,
    depth: int,
    lam: float,
    eps: float,
    min_support: float,
    use_heap: bool,
    rule_list_mode: bool,
    max_cache_size: int,
):
    print(f"\nBranching cost: {branching_cost}")
    print("  Running with falling constraint ON...")
    with_constraint = run_rashomon_set(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        branching_cost=branching_cost,
        depth=depth,
        lam=lam,
        eps=eps,
        min_support=min_support,
        enable_falling_constraint=True,
        use_heap=use_heap,
        rule_list_mode=rule_list_mode,
        max_cache_size=max_cache_size,
    )

    print("  Running with falling constraint OFF...")
    without_constraint = run_rashomon_set(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        branching_cost=branching_cost,
        depth=depth,
        lam=lam,
        eps=eps,
        min_support=min_support,
        enable_falling_constraint=False,
        use_heap=use_heap,
        rule_list_mode=rule_list_mode,
        max_cache_size=max_cache_size,
        budget_override=with_constraint["budget"],
    )

    return with_constraint, without_constraint


def _parse_branching_costs(costs_str: str):
    if costs_str is None:
        return [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    costs_str = costs_str.strip()
    if "," in costs_str:
        return [float(x.strip()) for x in costs_str.split(",")]
    return [float(costs_str)]


def expected_decision_sparsity_by_class_falling_tree(
    tree, X: np.ndarray, y: np.ndarray, class_label: int
):
    class_mask = y == class_label
    if np.sum(class_mask) == 0:
        return 0.0
    return expected_decision_sparsity_falling_tree(tree, X[class_mask])


def evaluate_tree(node, x, threshold=0.5):
    """Evaluate a tree node on a single sample."""
    if isinstance(node, Leaf):
        return 1 if node.pred_prob >= threshold else 0
    if x[node.feature] == 0:
        return evaluate_tree(node.left, x, threshold)
    return evaluate_tree(node.right, x, threshold)


def compute_tree_loss_by_class(tree, X, y, class_label, threshold=0.5):
    """Compute loss for a specific class (0 or 1) for a falling tree."""
    class_mask = y == class_label
    if np.sum(class_mask) == 0:
        return 0.0

    X_class = X[class_mask]
    y_class = y[class_mask]

    predictions = []
    for i in range(len(X_class)):
        pred = evaluate_tree(tree, X_class[i], threshold)
        predictions.append(pred)

    predictions = np.array(predictions)
    loss = np.mean(predictions != y_class)
    return loss

def main():
    parser = argparse.ArgumentParser(
        description="Compare Falling Trees Rashomon sets with falling constraint on vs off"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--label-column", type=str, default=None, help="Label column name")
    parser.add_argument("--num-estimators", type=int, default=200, help="GBDT estimators for binarization")

    parser.add_argument("--depth", type=int, default=5, help="Max depth of tree")
    parser.add_argument("--lam", type=float, default=0.005, help="Regularization parameter")
    parser.add_argument("--eps", type=float, default=0.02, help="Rashomon budget epsilon")
    parser.add_argument("--min-support", type=float, default=0.01, help="Minimum support for splits")
    parser.add_argument("--rule-list-mode", action="store_true", default=False, help="Use rule list mode")
    parser.add_argument("--use-heap", action="store_true", default=True, help="Use heap in R-set search")
    parser.add_argument("--max-cache-size", type=int, default=10**7, help="Max subproblem cache size")

    parser.add_argument(
        "--branching-costs",
        type=str,
        default=None,
        help="Single branching cost or comma-separated list",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of train/test splits to run in this process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="falling_trees_vs_regular_trees_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    branching_costs = _parse_branching_costs(args.branching_costs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y, label_column = load_and_binarize_dataset(
        args.dataset, args.label_column, args.num_estimators
    )

    dataset_name = Path(args.dataset).stem
    all_results = []
    detailed_rows = []

    print(f"Running {args.num_trials} train/test splits in this run.")
    split_indices = range(args.num_trials)

    for split_idx in split_indices:
        print(f"\n{'='*80}")
        print(f"Split {split_idx + 1}/{args.num_trials}")
        print(f"{'='*80}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split_idx, stratify=y
        )

        for bc in branching_costs:
            try:
                with_constraint, without_constraint = run_comparison_for_branching_cost(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    branching_cost=bc,
                    depth=args.depth,
                    lam=args.lam,
                    eps=args.eps,
                    min_support=args.min_support,
                    use_heap=args.use_heap,
                    rule_list_mode=args.rule_list_mode,
                    max_cache_size=args.max_cache_size,
                )

                result = {
                    "dataset": dataset_name,
                    "split_idx": split_idx,
                    "branching_cost": bc,
                    "with_constraint": with_constraint,
                    "without_constraint": without_constraint,
                }
                all_results.append(result)

                bc_str = str(bc).replace(".", "_")
                with open(
                    output_dir / f"{dataset_name}_split_{split_idx}_bc_{bc_str}_results.pkl",
                    "wb",
                ) as f:
                    pickle.dump(result, f)

                with_terms_mean = (
                    float(np.mean(with_constraint["terms"]))
                    if len(with_constraint["terms"]) > 0
                    else 0.0
                )
                without_terms_mean = (
                    float(np.mean(without_constraint["terms"]))
                    if len(without_constraint["terms"]) > 0
                    else 0.0
                )
                with_terms_pos_mean = (
                    float(np.mean(with_constraint["terms_pos"]))
                    if len(with_constraint["terms_pos"]) > 0
                    else 0.0
                )
                with_terms_neg_mean = (
                    float(np.mean(with_constraint["terms_neg"]))
                    if len(with_constraint["terms_neg"]) > 0
                    else 0.0
                )
                without_terms_pos_mean = (
                    float(np.mean(without_constraint["terms_pos"]))
                    if len(without_constraint["terms_pos"]) > 0
                    else 0.0
                )
                without_terms_neg_mean = (
                    float(np.mean(without_constraint["terms_neg"]))
                    if len(without_constraint["terms_neg"]) > 0
                    else 0.0
                )
                with_loss_pos_mean = (
                    float(np.mean(with_constraint["loss_pos"]))
                    if len(with_constraint["loss_pos"]) > 0
                    else 0.0
                )
                with_loss_neg_mean = (
                    float(np.mean(with_constraint["loss_neg"]))
                    if len(with_constraint["loss_neg"]) > 0
                    else 0.0
                )
                without_loss_pos_mean = (
                    float(np.mean(without_constraint["loss_pos"]))
                    if len(without_constraint["loss_pos"]) > 0
                    else 0.0
                )
                without_loss_neg_mean = (
                    float(np.mean(without_constraint["loss_neg"]))
                    if len(without_constraint["loss_neg"]) > 0
                    else 0.0
                )

                detailed_rows.append(
                    {
                        "dataset": dataset_name,
                        "split_idx": split_idx,
                        "branching_cost": bc,
                        "with_constraint_rset_size": with_constraint["rset_size"],
                        "without_constraint_rset_size": without_constraint["rset_size"],
                        "with_constraint_time": with_constraint["total_time"],
                        "without_constraint_time": without_constraint["total_time"],
                        "with_constraint_best_loss": with_constraint["best_loss"],
                        "without_constraint_best_loss": without_constraint["best_loss"],
                        "with_constraint_sparsity_mean": with_terms_mean,
                        "without_constraint_sparsity_mean": without_terms_mean,
                        "with_constraint_sparsity_pos_mean": with_terms_pos_mean,
                        "with_constraint_sparsity_neg_mean": with_terms_neg_mean,
                        "without_constraint_sparsity_pos_mean": without_terms_pos_mean,
                        "without_constraint_sparsity_neg_mean": without_terms_neg_mean,
                        "with_constraint_loss_pos_mean": with_loss_pos_mean,
                        "with_constraint_loss_neg_mean": with_loss_neg_mean,
                        "without_constraint_loss_pos_mean": without_loss_pos_mean,
                        "without_constraint_loss_neg_mean": without_loss_neg_mean,
                    }
                )

            except Exception as e:
                print(f"Error with split={split_idx}, branching_cost={bc}: {e}")
                import traceback

                traceback.print_exc()
                continue

    detailed_df = pd.DataFrame(detailed_rows)
    full_detailed_path = output_dir / f"{dataset_name}_full_detailed_results.csv"

    # Save per-branching-cost detailed results (merge by split)
    for bc in branching_costs:
        bc_data = detailed_df[detailed_df["branching_cost"] == bc]
        if len(bc_data) == 0:
            continue
        bc_str = str(bc).replace(".", "_")
        bc_detailed_path = output_dir / f"{dataset_name}_bc_{bc_str}_detailed_results.csv"
        if bc_detailed_path.exists():
            try:
                existing_bc = pd.read_csv(bc_detailed_path)
                bc_data = pd.concat([existing_bc, bc_data], ignore_index=True)
                bc_data = bc_data.drop_duplicates(
                    subset=["dataset", "split_idx", "branching_cost"], keep="last"
                )
            except Exception as e:
                print(f"Warning: could not merge detailed results for bc={bc}: {e}")
        bc_data.to_csv(bc_detailed_path, index=False)

    # Rebuild full detailed results from per-bc files to avoid race conditions
    bc_detailed_files = sorted(
        output_dir.glob(f"{dataset_name}_bc_*_detailed_results.csv")
    )
    if len(bc_detailed_files) > 0:
        full_detailed_df = pd.concat(
            [pd.read_csv(p) for p in bc_detailed_files], ignore_index=True
        )
        full_detailed_df = full_detailed_df.drop_duplicates(
            subset=["dataset", "split_idx", "branching_cost"], keep="last"
        )
    else:
        full_detailed_df = detailed_df
    full_detailed_df.to_csv(full_detailed_path, index=False)

    summary_rows = []
    for bc in branching_costs:
        bc_str = str(bc).replace(".", "_")
        bc_detailed_path = output_dir / f"{dataset_name}_bc_{bc_str}_detailed_results.csv"
        if bc_detailed_path.exists():
            try:
                bc_data = pd.read_csv(bc_detailed_path)
            except Exception as e:
                print(f"Warning: could not read detailed results for bc={bc}: {e}")
                bc_data = pd.DataFrame()
        else:
            bc_data = detailed_df[detailed_df["branching_cost"] == bc]

        if len(bc_data) == 0:
            continue
        n_bc = len(bc_data)
        sqrt_n = np.sqrt(n_bc)

        summary_rows.append(
            {
                "branching_cost": bc,
                "with_constraint_sparsity_mean": bc_data[
                    "with_constraint_sparsity_mean"
                ].mean(),
                "with_constraint_sparsity_se": bc_data[
                    "with_constraint_sparsity_mean"
                ].std()
                / sqrt_n,
                "with_constraint_sparsity_pos_mean": bc_data[
                    "with_constraint_sparsity_pos_mean"
                ].mean(),
                "with_constraint_sparsity_pos_se": bc_data[
                    "with_constraint_sparsity_pos_mean"
                ].std()
                / sqrt_n,
                "with_constraint_sparsity_neg_mean": bc_data[
                    "with_constraint_sparsity_neg_mean"
                ].mean(),
                "with_constraint_sparsity_neg_se": bc_data[
                    "with_constraint_sparsity_neg_mean"
                ].std()
                / sqrt_n,
                "without_constraint_sparsity_mean": bc_data[
                    "without_constraint_sparsity_mean"
                ].mean(),
                "without_constraint_sparsity_se": bc_data[
                    "without_constraint_sparsity_mean"
                ].std()
                / sqrt_n,
                "without_constraint_sparsity_pos_mean": bc_data[
                    "without_constraint_sparsity_pos_mean"
                ].mean(),
                "without_constraint_sparsity_pos_se": bc_data[
                    "without_constraint_sparsity_pos_mean"
                ].std()
                / sqrt_n,
                "without_constraint_sparsity_neg_mean": bc_data[
                    "without_constraint_sparsity_neg_mean"
                ].mean(),
                "without_constraint_sparsity_neg_se": bc_data[
                    "without_constraint_sparsity_neg_mean"
                ].std()
                / sqrt_n,
                "with_constraint_rset_size_mean": bc_data[
                    "with_constraint_rset_size"
                ].mean(),
                "without_constraint_rset_size_mean": bc_data[
                    "without_constraint_rset_size"
                ].mean(),
                "with_constraint_time_mean": bc_data["with_constraint_time"].mean(),
                "with_constraint_time_se": bc_data["with_constraint_time"].std()
                / sqrt_n,
                "without_constraint_time_mean": bc_data["without_constraint_time"].mean(),
                "without_constraint_time_se": bc_data["without_constraint_time"].std()
                / sqrt_n,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Save per-branching-cost summary files (one row each) for aggregation
    for bc in branching_costs:
        bc_summary = summary_df[summary_df["branching_cost"] == bc]
        if len(bc_summary) == 0:
            continue
        bc_str = str(bc).replace(".", "_")
        bc_summary_path = output_dir / f"{dataset_name}_bc_{bc_str}_summary.csv"
        if bc_summary_path.exists():
            try:
                existing_bc_summary = pd.read_csv(bc_summary_path)
                bc_summary = pd.concat(
                    [existing_bc_summary, bc_summary], ignore_index=True
                )
                bc_summary = bc_summary.drop_duplicates(
                    subset=["branching_cost"], keep="last"
                )
            except Exception as e:
                print(f"Warning: could not merge summary for bc={bc}: {e}")
        bc_summary.to_csv(bc_summary_path, index=False)

    # Save full summary by combining per-bc summaries (avoids race conditions)
    summary_files = sorted(output_dir.glob(f"{dataset_name}_bc_*_summary.csv"))
    if len(summary_files) > 0:
        full_summary_df = pd.concat(
            [pd.read_csv(p) for p in summary_files], ignore_index=True
        )
        full_summary_df = full_summary_df.drop_duplicates(
            subset=["branching_cost"], keep="last"
        )
    else:
        full_summary_df = summary_df
    full_summary_path = output_dir / f"{dataset_name}_full_summary.csv"
    full_summary_df.to_csv(full_summary_path, index=False)

    print(f"\nSaved detailed results to {full_detailed_path}")
    print(f"Saved summary results to {full_summary_path}")


if __name__ == "__main__":
    main()
