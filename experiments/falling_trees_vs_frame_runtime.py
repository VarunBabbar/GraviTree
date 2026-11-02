"""
Compare Falling Trees vs FRAME with runtime tracking and branching cost sweep.
Tracks runtime, loss, and sparsity for both algorithms across different branching costs.
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
import pickle

# Add parent directory to path so we can import falling_trees and frame
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from falling_trees.binarize_dataset import binarize_dataset
from falling_trees import frl_rashomon_set_alg
from falling_trees.frl_rashomon_set_alg import (
    Node, Leaf, OptFallingTree, OptFallingRset, tree_obj, 
    _subproblem_optimal_objectives, normalized_colless_index
)
from frame.rashomon_sets import FRLRashomonSet
from utils import (
    expected_decision_sparsity_falling_tree,
    expected_decision_sparsity_frame_frl,
    compute_tree_test_loss_threshold,
    predict_proba_falling_tree,
)

def frl_accuracy(frl, X, y):
    """Compute accuracy of an FRL."""
    y_pred = frl.predict(X)
    return accuracy_score(y, y_pred)


def frl_loss(frl, X, y):
    """Compute loss (1 - accuracy) of an FRL."""
    return 1 - frl_accuracy(frl, X, y)


def run_comparison_for_branching_cost(
    X_train, y_train, X_test, y_test,
    branching_cost: float,
    num_estimators: int = 50,
    depth: int = 5,
    lam: float = 0.005,
    eps: float = 0.02,
    min_support: float = 0.02,
    enable_falling_constraint: bool = True,
    rule_list_mode: bool = False,
    use_heap: bool = True,
    max_cache_size: int = 10**7,
    frame_epsilon: float = None,
    frame_curiosity_func: str = 'ucb+',
    split_idx: int = 0,
    max_len: int = 2,
    threshold: float = 0.5,
):
    """
    Run comparison for a single branching cost value.
    Returns results dictionary with runtime, loss, and sparsity metrics.
    """
    if frame_epsilon is None:
        frame_epsilon = eps
    
    print(f"\n{'='*60}")
    print(f"Split {split_idx + 1}, Branching Cost: {branching_cost}")
    print(f"{'='*60}")
    
    features = list(range(X_train.shape[1]))
    n_train = X_train.shape[0]
    row_idx_train = np.arange(n_train)
    X_train_bool = X_train.astype(bool).copy()  # Make a copy to avoid modifying original
    X_test_bool = X_test.astype(bool).copy()   # Make a copy to avoid modifying original
    
    # Set cache size and clear cache
    frl_rashomon_set_alg.MAX_CACHE_SIZE = max_cache_size
    _subproblem_optimal_objectives.clear()
    
    kwargs = {
        "branching_cost": branching_cost,
        "min_support": min_support
    }
    
    # ========== Falling Trees ==========
    print(f"\nRunning OptFallingTree...")
    start_time = time.time()
    best_loss, best_tree, pmax, depth_of_pmax = OptFallingTree(
        X_train.values,
        y_train.values,
        row_idx_train,
        depth,
        lam,
        features,
        n=n_train,
        enable_falling_constraint=enable_falling_constraint,
        rule_list_mode=rule_list_mode,
        **kwargs
    )
    opt_tree_time = time.time() - start_time
    
    # ========== FRAME ==========
    print(f"\nRunning FRAME...")
    start_time = time.time()
    frame_rset = FRLRashomonSet(epsilon=frame_epsilon)
    frame_rset.fit(X_train_bool, y_train, curiosity_func=frame_curiosity_func, max_len=max_len)
    frame_time = time.time() - start_time
    
    print(f"Found {len(frame_rset.rset)} FRLs in FRAME Rashomon set")
    print(f"FRAME total time: {frame_time:.2f} seconds")
    
    # After fit(), X_train_bool has been modified in place to include complement features
    # We need to expand X_test_bool to match the same structure for prediction
    # Check if FRAME added complement features
    if hasattr(frame_rset.reference_model, 'included_complement') and frame_rset.reference_model.included_complement:
        # Expand X_test_bool to match X_train_bool structure
        n_original_features = X_test_bool.shape[1]
        for col in X_test_bool.columns:
            X_test_bool['~' + col] = ~X_test_bool[col]
        print(f"Expanded X_test_bool from {n_original_features} to {X_test_bool.shape[1]} features to match training data")
    
    # CRITICAL: Ensure X_test_bool columns match the order of frl.features
    # The rule_list uses numeric indices that refer to positions in frl.features
    # So the column order must match exactly
    reference_features = frame_rset.reference_model.features
    # Reorder X_test_bool columns to match reference_features order
    # Add any missing columns (shouldn't happen, but just in case)
    for feat in reference_features:
        if feat not in X_test_bool.columns:
            print(f"Warning: Feature {feat} not found in X_test_bool, adding zeros")
            X_test_bool[feat] = False
    # Reorder to match reference_features exactly
    X_test_bool = X_test_bool[reference_features]
    
    # Compute metrics for FRAME (on test set for sparsity, both train and test for loss)
    # and ROC curves
    terms_frame = [expected_decision_sparsity_frame_frl(frl, X_test.astype(bool)) for frl in frame_rset.rset]
    loss_frame_train = [frl_loss(frl, X_train_bool, y_train) for frl in frame_rset.rset]
    loss_frame_test = [frl_loss(frl, X_test_bool, y_test) for frl in frame_rset.rset]
    roc_frame = []
    for frl in frame_rset.rset:
        # Ensure X_test_bool columns match this frl's features order
        # Reorder to match frl.features (should be same as reference_features, but be safe)
        X_test_bool_reordered = X_test_bool[frl.features] if hasattr(frl, 'features') else X_test_bool
        frl_probs = frl.predict_proba(X_test_bool_reordered)
        fpr, tpr, _ = roc_curve(y_test.values, frl_probs)
        roc_frame.append({'fpr': fpr, 'tpr': tpr})
    
    # Compute decision sparsity and loss by class for FRAME
    # For decision sparsity, use original X (function will expand if needed)
    # For loss, use expanded X_train_bool and X_test_bool
    # Train
    terms_frame_train_pos = [compute_decision_sparsity_by_class_frame_frl(frl, X_train.astype(bool).values, y_train.values, 1) for frl in frame_rset.rset]
    terms_frame_train_neg = [compute_decision_sparsity_by_class_frame_frl(frl, X_train.astype(bool).values, y_train.values, 0) for frl in frame_rset.rset]
    loss_frame_train_pos = [compute_frl_loss_by_class(frl, X_train_bool.values, y_train.values, 1) for frl in frame_rset.rset]
    loss_frame_train_neg = [compute_frl_loss_by_class(frl, X_train_bool.values, y_train.values, 0) for frl in frame_rset.rset]
    # Test
    terms_frame_test_pos = [compute_decision_sparsity_by_class_frame_frl(frl, X_test.astype(bool).values, y_test.values, 1) for frl in frame_rset.rset]
    terms_frame_test_neg = [compute_decision_sparsity_by_class_frame_frl(frl, X_test.astype(bool).values, y_test.values, 0) for frl in frame_rset.rset]
    loss_frame_test_pos = [compute_frl_loss_by_class(frl, X_test_bool.values, y_test.values, 1) for frl in frame_rset.rset]
    loss_frame_test_neg = [compute_frl_loss_by_class(frl, X_test_bool.values, y_test.values, 0) for frl in frame_rset.rset]
    
    frame_best_loss_train = min(loss_frame_train) if len(loss_frame_train) > 0 else float("inf")
    aligned_eps = max(0.0, (frame_best_loss_train + eps) - best_loss)
    B = best_loss + aligned_eps
    print(f"Best train loss: {best_loss:.4f}, FRAME best: {frame_best_loss_train:.4f}, "
          f"Aligned epsilon: {aligned_eps:.4f}, Budget: {B:.4f}")

    print(f"\nRunning OptFallingRset...")
    start_time = time.time()
    R = OptFallingRset(
        X_train.values,
        y_train.values,
        row_idx_train,
        depth,
        lam,
        B=B,
        features=features,
        n=n_train,
        enable_falling_constraint=enable_falling_constraint,
        use_heap=use_heap,
        rule_list_mode=rule_list_mode,
        **kwargs
    )
    rset_time = time.time() - start_time
    falling_trees_total_time = opt_tree_time + rset_time
    
    print(f"Found {len(R)} trees in Rashomon set")
    print(f"Falling Trees total time: {falling_trees_total_time:.2f} seconds")
    
    # Compute metrics for Falling Trees (on test set)
    terms_falling_trees = [expected_decision_sparsity_falling_tree(model[0], X_test.astype(bool).values) for model in R]
    loss_falling_trees_train = [tree_obj(model[0]) for model in R]
    # Compute test loss and ROC curves for falling trees
    loss_falling_trees_test = []
    roc_falling_trees = []
    for model in R:
        tree = model[0]
        test_loss = compute_tree_test_loss_threshold(tree, X_test.values, y_test.values, threshold)
        loss_falling_trees_test.append(test_loss)
        # ROC curve (use predicted probabilities)
        tree_probs = predict_proba_falling_tree(tree, X_test.values)
        fpr, tpr, _ = roc_curve(y_test.values, tree_probs)
        roc_falling_trees.append({'fpr': fpr, 'tpr': tpr})
    
    # Compute decision sparsity and loss by class for Falling Trees
    # Train
    terms_falling_trees_train_pos = [compute_decision_sparsity_by_class_falling_tree(model[0], X_train.astype(bool).values, y_train.values, 1) for model in R]
    terms_falling_trees_train_neg = [compute_decision_sparsity_by_class_falling_tree(model[0], X_train.astype(bool).values, y_train.values, 0) for model in R]
    loss_falling_trees_train_pos = [compute_tree_loss_by_class(model[0], X_train.values, y_train.values, 1) for model in R]
    loss_falling_trees_train_neg = [compute_tree_loss_by_class(model[0], X_train.values, y_train.values, 0) for model in R]
    # Test
    terms_falling_trees_test_pos = [compute_decision_sparsity_by_class_falling_tree(model[0], X_test.astype(bool).values, y_test.values, 1) for model in R]
    terms_falling_trees_test_neg = [compute_decision_sparsity_by_class_falling_tree(model[0], X_test.astype(bool).values, y_test.values, 0) for model in R]
    loss_falling_trees_test_pos = [compute_tree_loss_by_class(model[0], X_test.values, y_test.values, 1) for model in R]
    loss_falling_trees_test_neg = [compute_tree_loss_by_class(model[0], X_test.values, y_test.values, 0) for model in R]

    # Return results
    results = {
        'split_idx': split_idx,
        'branching_cost': branching_cost,
        'falling_trees_time': falling_trees_total_time,
        'frame_time': frame_time,
        'falling_trees_rset_size': len(R),
        'frame_rset_size': len(frame_rset.rset),
        'terms_falling_trees': terms_falling_trees,
        'loss_falling_trees_train': loss_falling_trees_train,
        'loss_falling_trees_test': loss_falling_trees_test,
        'terms_frame': terms_frame,
        'loss_frame_train': loss_frame_train,
        'loss_frame_test': loss_frame_test,
        # ROC curve data (per model in Rashomon set, on test set)
        'roc_falling_trees': roc_falling_trees,
        'roc_frame': roc_frame,
        # By class metrics for Falling Trees
        'terms_falling_trees_train_pos': terms_falling_trees_train_pos,
        'terms_falling_trees_train_neg': terms_falling_trees_train_neg,
        'loss_falling_trees_train_pos': loss_falling_trees_train_pos,
        'loss_falling_trees_train_neg': loss_falling_trees_train_neg,
        'terms_falling_trees_test_pos': terms_falling_trees_test_pos,
        'terms_falling_trees_test_neg': terms_falling_trees_test_neg,
        'loss_falling_trees_test_pos': loss_falling_trees_test_pos,
        'loss_falling_trees_test_neg': loss_falling_trees_test_neg,
        # By class metrics for FRAME
        'terms_frame_train_pos': terms_frame_train_pos,
        'terms_frame_train_neg': terms_frame_train_neg,
        'loss_frame_train_pos': loss_frame_train_pos,
        'loss_frame_train_neg': loss_frame_train_neg,
        'terms_frame_test_pos': terms_frame_test_pos,
        'terms_frame_test_neg': terms_frame_test_neg,
        'loss_frame_test_pos': loss_frame_test_pos,
        'loss_frame_test_neg': loss_frame_test_neg,
    }
    
    return results


def compute_tree_test_loss(tree, X_test, y_test):
    """Compute test loss for a falling tree."""
    n_test = X_test.shape[0]
    predictions = []
    
    for i in range(n_test):
        pred = evaluate_tree(tree, X_test[i])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    loss = np.mean(predictions != y_test)
    return loss


def evaluate_tree(node, x, threshold=0.5):
    """Evaluate a tree node on a single sample."""
    if isinstance(node, Leaf):
        return 1 if node.pred_prob >= threshold else 0
    else:
        if x[node.feature] == 0:
            return evaluate_tree(node.left, x)
        else:
            return evaluate_tree(node.right, x)


def compute_tree_loss_by_class(tree, X, y, class_label, threshold=0.5):
    """Compute loss for a specific class (0 or 1) for a falling tree."""
    class_mask = (y == class_label)
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


def compute_decision_sparsity_by_class_falling_tree(tree, X, y, class_label):
    """Compute decision sparsity for a specific class (0 or 1) for a falling tree."""
    from utils import expected_decision_sparsity_falling_tree
    
    class_mask = (y == class_label)
    if np.sum(class_mask) == 0:
        return 0.0
    
    X_class = X[class_mask]
    return expected_decision_sparsity_falling_tree(tree, X_class)


def compute_frl_loss_by_class(frl, X, y, class_label):
    """Compute loss for a specific class (0 or 1) for a FRAME FRL."""
    class_mask = (y == class_label)
    if np.sum(class_mask) == 0:
        return 0.0
    
    X_class = X[class_mask]
    y_class = y[class_mask]
    
    y_pred = frl.predict(X_class)
    loss = np.mean(y_pred != y_class)
    return loss


def compute_decision_sparsity_by_class_frame_frl(frl, X: pd.DataFrame, y: pd.Series, class_label: int):
    """Compute decision sparsity for a specific class (0 or 1) for a FRAME FRL."""
    
    class_mask = (y == class_label)
    if np.sum(class_mask) == 0:
        return 0.0
    
    X_class = X[class_mask]
    return expected_decision_sparsity_frame_frl(frl, X_class)


def create_runtime_plot(summary_df, dataset_name, output_dir):
    """Create plot showing runtime vs branching cost."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot Falling Trees runtime
    ax.plot(
        summary_df['branching_cost'],
        summary_df['falling_trees_time'],
        marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue'
    )
    
    # Plot FRAME runtime (constant line)
    if len(summary_df) > 0:
        frame_time_mean = summary_df['frame_time'].mean()
        ax.axhline(
            y=frame_time_mean,
            color='red', linestyle='--', linewidth=2, label=f'FRAME (mean: {frame_time_mean:.2f}s)'
        )
    
    ax.set_xlabel('Branching Cost', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title(f'Runtime Comparison: Falling Trees vs FRAME\n{dataset_name}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved runtime plot to {output_dir / f'{dataset_name}_runtime_comparison.png'}")


def create_scatterplots(all_results, dataset_name, output_dir):
    """Create scatterplots of loss vs sparsity for each branching cost."""
    n_bc = len(all_results)
    
    # Determine grid layout
    if n_bc <= 3:
        n_cols = n_bc
        n_rows = 1
    elif n_bc <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_bc + 2) // 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, results in enumerate(all_results):
        ax = axes[idx]
        bc = results['branching_cost']
        
        # Scatter plot: loss vs sparsity
        ax.scatter(
            results['terms_falling_trees'],
            results['loss_falling_trees'],
            color='blue', label='Falling Trees', alpha=0.6, s=50
        )
        ax.scatter(
            results['terms_frame'],
            results['loss_frame'],
            color='red', label='FRAME', alpha=0.6, s=50, marker='^'
        )
        
        ax.set_xlabel('Sparsity (Number of Terms)', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Loss vs Sparsity\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_loss_vs_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved scatterplots to {output_dir / f'{dataset_name}_loss_vs_sparsity.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Falling Trees vs FRAME with runtime tracking')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset CSV file')
    parser.add_argument('--label-column', type=str, default=None,
                        help='Name of label column (default: last column)')
    parser.add_argument('--num-estimators', type=int, default=50,
                        help='Number of estimators for binarization (default: 50)')
    
    # Algorithm parameters
    parser.add_argument('--depth', type=int, default=5,
                        help='Maximum depth of tree (default: 5)')
    parser.add_argument('--lam', type=float, default=0.005,
                        help='Regularization parameter lambda (default: 0.005)')
    parser.add_argument('--eps', type=float, default=0.02,
                        help='Epsilon for Rashomon set budget (default: 0.02)')
    parser.add_argument('--min-support', type=float, default=0.02,
                        help='Minimum support for splits (default: 0.02)')
    parser.add_argument('--enable-falling-constraint', action='store_true', default=True,
                        help='Enable falling tree constraint (default: True)')
    parser.add_argument('--disable-falling-constraint', dest='enable_falling_constraint', action='store_false',
                        help='Disable falling tree constraint')
    parser.add_argument('--rule-list-mode', action='store_true', default=False,
                        help='Use rule list mode (default: False)')
    parser.add_argument('--use-heap', action='store_true', default=True,
                        help='Use heap for storing trees (default: True)')
    parser.add_argument('--max-len', type=int, default=2,
                        help='Maximum length of antecedent (default: 2)')
    parser.add_argument('--num-trials', type=int, default=5,
                        help='Number of trials (default: 5)')
    
    # Branching costs
    parser.add_argument('--branching-costs', type=str, default=None,
                        help='Single branching cost or comma-separated list (default: 0.0,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1)')
    
    # FRAME parameters
    parser.add_argument('--frame-epsilon', type=float, default=None,
                        help='Epsilon for FRAME (default: same as --eps)')
    parser.add_argument('--frame-curiosity', type=str, default='ucb+',
                        choices=['ucb+', 'ucb-'],
                        help='Curiosity function for FRAME (default: ucb+)')
    
    # Cache settings
    parser.add_argument('--max-cache-size', type=int, default=10**7,
                        help='Maximum size of subproblem cache (default: 10^7)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='falling_trees_vs_frame_runtime_results',
                        help='Output directory for results (default: falling_trees_vs_frame_runtime_results)')
    
    args = parser.parse_args()
    
    # Parse branching costs
    if args.branching_costs is None:
        branching_costs = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    else:
        # Handle both single value and comma-separated list
        costs_str = args.branching_costs.strip()
        if ',' in costs_str:
            branching_costs = [float(x.strip()) for x in costs_str.split(',')]
        else:
            # Single value
            branching_costs = [float(costs_str)]
    df = pd.read_csv(args.dataset)
    output_dir = Path(args.output_dir)
    
    # Load and binarize dataset once
    print(f"\nLoading and binarizing dataset...")
    
    dataset_name = Path(args.dataset).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run n_splits train/test splits
    n_splits = args.num_trials
    all_results = []
    
    for split_idx in range(n_splits):
        if args.label_column is None:
            label_column = df.columns[-1]
        else:
            label_column = args.label_column
        X,y = df.drop(columns=[label_column]), df[label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_idx, stratify=y)
        X_train, thresholds, header, threshold_guess_time = binarize_dataset(X_train, num_estimators=args.num_estimators)
        X_test = binarize_dataset(X_test, num_estimators=args.num_estimators, thresholds=thresholds, header=header)
        print(f"\n{'='*80}")
        print(f"Split {split_idx + 1}/{n_splits}")
        print(f"{'='*80}")
        
        # Run for each branching cost
        for bc in branching_costs:
            try:
                results = run_comparison_for_branching_cost(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    branching_cost=bc,
                    num_estimators=args.num_estimators,
                    depth=args.depth,
                    lam=args.lam,
                    eps=args.eps,
                    min_support=args.min_support,
                    enable_falling_constraint=args.enable_falling_constraint,
                    rule_list_mode=args.rule_list_mode,
                    use_heap=args.use_heap,
                    max_cache_size=args.max_cache_size,
                    frame_epsilon=args.frame_epsilon,
                    frame_curiosity_func=args.frame_curiosity,
                    split_idx=split_idx,
                    max_len=args.max_len,
                )
                all_results.append(results)
                
                # Save individual result
                bc_str = str(bc).replace('.', '_')
                
                with open(output_dir / f'{dataset_name}_split_{split_idx}_bc_{bc_str}_results.pkl', 'wb') as f:
                    pickle.dump(results, f)
                
            except Exception as e:
                print(f"Error with split={split_idx}, branching_cost={bc}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate results and save CSV
    summary_data = []
    for r in all_results:
        # Compute means across models in Rashomon set for this split
        ft_loss_train_mean = np.mean(r['loss_falling_trees_train'])
        ft_loss_test_mean = np.mean(r['loss_falling_trees_test'])
        frame_loss_train_mean = np.mean(r['loss_frame_train'])
        frame_loss_test_mean = np.mean(r['loss_frame_test'])
        ft_sparsity_mean = np.mean(r['terms_falling_trees']) if len(r['terms_falling_trees']) > 0 else 0
        frame_sparsity_mean = np.mean(r['terms_frame']) if len(r['terms_frame']) > 0 else 0
        
        summary_data.append({
            'split_idx': r['split_idx'],
            'branching_cost': r['branching_cost'],
            'falling_trees_time': r['falling_trees_time'],
            'frame_time': r['frame_time'],
            'falling_trees_rset_size': r['falling_trees_rset_size'],
            'frame_rset_size': r['frame_rset_size'],
            # Loss means (across models in Rashomon set)
            'falling_trees_loss_train_mean': ft_loss_train_mean,
            'falling_trees_loss_test_mean': ft_loss_test_mean,
            'frame_loss_train_mean': frame_loss_train_mean,
            'frame_loss_test_mean': frame_loss_test_mean,
            # Decision sparsity means (across models in Rashomon set)
            'falling_trees_sparsity_mean': ft_sparsity_mean,
            'frame_sparsity_mean': frame_sparsity_mean,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary CSV grouped by branching cost (with mean and standard error over splits)
    n_splits = len(summary_df['split_idx'].unique()) if 'split_idx' in summary_df.columns else 5
    sqrt_n = np.sqrt(n_splits)
    
    for bc in branching_costs:
        bc_str = str(bc).replace('.', '_')
        bc_data = summary_df[summary_df['branching_cost'] == bc]
        if len(bc_data) > 0:
            # Compute mean and standard error (std / sqrt(n)) over splits
            n_bc_splits = len(bc_data)
            sqrt_n_bc = np.sqrt(n_bc_splits)
            
            avg_data = {
                'branching_cost': bc,
                # Runtime: mean and standard error
                'falling_trees_time_mean': bc_data['falling_trees_time'].mean(),
                'falling_trees_time_se': bc_data['falling_trees_time'].std() / sqrt_n_bc,
                'frame_time_mean': bc_data['frame_time'].mean(),
                'frame_time_se': bc_data['frame_time'].std() / sqrt_n_bc,
                # Loss: mean and standard error
                'falling_trees_loss_train_mean': bc_data['falling_trees_loss_train_mean'].mean(),
                'falling_trees_loss_train_se': bc_data['falling_trees_loss_train_mean'].std() / sqrt_n_bc,
                'falling_trees_loss_test_mean': bc_data['falling_trees_loss_test_mean'].mean(),
                'falling_trees_loss_test_se': bc_data['falling_trees_loss_test_mean'].std() / sqrt_n_bc,
                'frame_loss_train_mean': bc_data['frame_loss_train_mean'].mean(),
                'frame_loss_train_se': bc_data['frame_loss_train_mean'].std() / sqrt_n_bc,
                'frame_loss_test_mean': bc_data['frame_loss_test_mean'].mean(),
                'frame_loss_test_se': bc_data['frame_loss_test_mean'].std() / sqrt_n_bc,
                # Decision sparsity: mean and standard error
                'falling_trees_sparsity_mean': bc_data['falling_trees_sparsity_mean'].mean(),
                'falling_trees_sparsity_se': bc_data['falling_trees_sparsity_mean'].std() / sqrt_n_bc,
                'frame_sparsity_mean': bc_data['frame_sparsity_mean'].mean(),
                'frame_sparsity_se': bc_data['frame_sparsity_mean'].std() / sqrt_n_bc,
                # Rashomon set sizes
                'falling_trees_rset_size_mean': bc_data['falling_trees_rset_size'].mean(),
                'frame_rset_size_mean': bc_data['frame_rset_size'].mean(),
            }
            avg_df = pd.DataFrame([avg_data])
            avg_df.to_csv(output_dir / f'{dataset_name}_bc_{bc_str}_summary.csv', index=False)
    
    # Also save full summary
    summary_df.to_csv(output_dir / f'{dataset_name}_full_summary.csv', index=False)
    
    print(f"\nResults saved to {output_dir}")

