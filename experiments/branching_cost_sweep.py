"""
Script to sweep over branching costs, compute R-sets, and analyze colless indices.
Runs over multiple train/test splits and computes mean + std dev of metrics.

Usage:
    python branching_cost_sweep.py --dataset <dataset_name> [--branching_cost <cost>] [--depth <depth>] [--lam <lam>] [--eps <eps>] [--min_support <support>]

If branching_cost is not specified, sweeps over default range.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import frl_rashomon_set_alg
sys.path.insert(0, str(Path(__file__).parent.parent / 'falling_trees'))
from frl_rashomon_set_alg import (
    OptFallingTree, OptFallingRset, binarize_dataset,
    normalized_colless_index, _subproblem_optimal_objectives,
    MAX_CACHE_SIZE
)

# Default configuration
DATA_DIR = Path(__file__).parent.parent / 'data' / 'benchmark'
DEFAULT_DEPTH = 5
DEFAULT_MIN_SUPPORT = 0.02
DEFAULT_LAM = 0.005
DEFAULT_EPS = 0.02
N_SPLITS = 5
RANDOM_STATE_BASE = 42

# Default branching costs to sweep over (if not specified)
DEFAULT_BRANCHING_COSTS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]

# Get all CSV files in benchmark directory
def get_datasets():
    """Get all CSV dataset files from the benchmark directory."""
    datasets = []
    for csv_file in sorted(DATA_DIR.glob('*.csv')):
        datasets.append(csv_file)
    return datasets

def load_and_preprocess_dataset(dataset_path):
    """Load and preprocess a dataset."""
    print(f"\nLoading dataset: {dataset_path.name}")
    df = pd.read_csv(dataset_path)
    
    # Check if data needs binarization
    # Assume if all values are already 0/1, no binarization needed
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]
    
    # Check if features are already binary
    X_features = df[feature_cols]
    is_binary = True
    for col in feature_cols:
        unique_vals = X_features[col].unique()
        if not all(val in [0, 1] for val in unique_vals if pd.notna(val)):
            is_binary = False
            break
    
    if not is_binary:
        print(f"  Binarizing dataset...")
        df, thresholds, header, threshold_guess_time = binarize_dataset(df, num_estimators=150)
    
    # Convert to numpy arrays
    X = df.iloc[:, :-1].values.astype(int)
    y = df.iloc[:, -1].values.astype(int)
    
    # Ensure label is binary (0 or 1)
    if not all(val in [0, 1] for val in np.unique(y)):
        print(f"  Warning: Label column contains non-binary values. Converting...")
        y = (y > 0).astype(int)
    
    print(f"  Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive label proportion: {y.mean():.3f}")
    
    return X, y

def compute_rset_for_branching_cost(X_train, y_train, branching_cost, depth, lam, eps, min_support):
    """Compute R-set for a given branching cost."""
    # Clear cache for each run
    _subproblem_optimal_objectives.clear()
    
    n = X_train.shape[0]
    features = list(range(X_train.shape[1]))
    row_idx = np.arange(n)
    
    kwargs = {
        "branching_cost": branching_cost,
        "min_support": min_support
    }
    
    # First, find optimal tree to get best_loss
    start_time = time.time()
    best_loss, best_tree, pmax, depth_of_pmax = OptFallingTree(
        X_train, y_train, row_idx, depth, lam, features, n=n,
        enable_falling_constraint=True,
        rule_list_mode=False,
        **kwargs
    )
    opt_tree_time = time.time() - start_time
    
    # Compute R-set with budget = best_loss * (1 + eps)
    start_time = time.time()
    R = OptFallingRset(
        X_train, y_train, row_idx, depth, lam,
        B=best_loss * (1 + eps),
        features=features, n=n,
        enable_falling_constraint=True,
        use_heap=True,
        rule_list_mode=False,
        **kwargs
    )
    rset_time = time.time() - start_time
    
    total_time = opt_tree_time + rset_time
    
    # Compute colless indices for all trees in R-set
    colless_indices = []
    if len(R) > 0:
        zeroth_element = R[0]
        if len(zeroth_element) == 5:
            for (tree, p_max, depth_of_pmax_leaf, obj, profile) in R:
                colless_idx = normalized_colless_index(tree)
                colless_indices.append(colless_idx)
        else:
            for (tree, p_max, depth_of_pmax_leaf, obj) in R:
                colless_idx = normalized_colless_index(tree)
                colless_indices.append(colless_idx)
    
    return {
        'rset_size': len(R),
        'best_loss': best_loss,
        'colless_indices': colless_indices,
        'opt_tree_time': opt_tree_time,
        'rset_time': rset_time,
        'total_time': total_time,
        'mean_colless': np.mean(colless_indices) if colless_indices else 0.0,
        'std_colless': np.std(colless_indices) if colless_indices else 0.0,
        'min_colless': np.min(colless_indices) if colless_indices else 0.0,
        'max_colless': np.max(colless_indices) if colless_indices else 0.0,
    }

def run_experiment(dataset_path, output_dir, branching_costs, depth, lam, eps, min_support):
    """Run the full experiment for one dataset with specified hyperparameters."""
    dataset_name = dataset_path.stem
    
    print(f"\n{'='*80}")
    print(f"Running experiment for dataset: {dataset_name}")
    print(f"Hyperparameters: depth={depth}, lam={lam}, eps={eps}, min_support={min_support}")
    print(f"Branching costs: {branching_costs}")
    print(f"{'='*80}")
    
    # Load and preprocess dataset
    X, y = load_and_preprocess_dataset(dataset_path)
    
    # Store results for all splits and branching costs
    results = []
    
    # Run over multiple train/test splits
    for split_idx in range(N_SPLITS):
        print(f"\n--- Split {split_idx + 1}/{N_SPLITS} ---")
        
        # Create train/test split (use 80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE_BASE + split_idx, stratify=y
        )
        
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # Sweep over branching costs
        for branching_cost in tqdm(branching_costs, desc=f"  Branching costs (split {split_idx + 1})"):
            try:
                result = compute_rset_for_branching_cost(
                    X_train, y_train, branching_cost, depth, lam, eps, min_support
                )
                
                result['split_idx'] = split_idx
                result['branching_cost'] = branching_cost
                result['dataset'] = dataset_name
                result['depth'] = depth
                result['lam'] = lam
                result['eps'] = eps
                result['min_support'] = min_support
                
                results.append(result)
                
            except Exception as e:
                print(f"    Error with branching_cost={branching_cost}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print(f"  Warning: No results collected for {dataset_name}")
        return None
    
    # Aggregate results: compute mean and std across splits for each branching cost
    aggregated = []
    for bc in branching_costs:
        bc_results = results_df[results_df['branching_cost'] == bc]
        if len(bc_results) == 0:
            continue
        
        # Aggregate metrics
        agg_result = {
            'branching_cost': bc,
            'dataset': dataset_name,
            'depth': depth,
            'lam': lam,
            'eps': eps,
            'min_support': min_support,
            'mean_rset_size': bc_results['rset_size'].mean(),
            'std_rset_size': bc_results['rset_size'].std(),
            'mean_best_loss': bc_results['best_loss'].mean(),
            'std_best_loss': bc_results['best_loss'].std(),
            'mean_total_time': bc_results['total_time'].mean(),
            'std_total_time': bc_results['total_time'].std(),
            'mean_mean_colless': bc_results['mean_colless'].mean(),
            'std_mean_colless': bc_results['mean_colless'].std(),
            'mean_std_colless': bc_results['std_colless'].mean(),
            'mean_min_colless': bc_results['min_colless'].mean(),
            'mean_max_colless': bc_results['max_colless'].mean(),
        }
        
        # Also aggregate all colless indices across all splits
        all_colless = []
        for colless_list in bc_results['colless_indices']:
            all_colless.extend(colless_list)
        
        if all_colless:
            agg_result['overall_mean_colless'] = np.mean(all_colless)
            agg_result['overall_std_colless'] = np.std(all_colless)
        else:
            agg_result['overall_mean_colless'] = 0.0
            agg_result['overall_std_colless'] = 0.0
        
        aggregated.append(agg_result)
    
    aggregated_df = pd.DataFrame(aggregated)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For array jobs (single branching_cost), save with unique filename
    # For full sweeps (multiple branching_costs), save with dataset name
    if len(branching_costs) == 1:
        # Single branching_cost: save with unique filename to avoid overwrites
        bc_str = str(branching_costs[0]).replace('.', '_')
        results_df.to_csv(output_dir / f'{dataset_name}_bc_{bc_str}_detailed_results.csv', index=False)
        aggregated_df.to_csv(output_dir / f'{dataset_name}_bc_{bc_str}_aggregated_results.csv', index=False)
        print(f"  Skipping plots (single branching_cost). Run aggregate_and_plot_results.py after all jobs complete.")
    else:
        # Multiple branching_costs: save with dataset name and create plots
        results_df.to_csv(output_dir / f'{dataset_name}_detailed_results.csv', index=False)
        aggregated_df.to_csv(output_dir / f'{dataset_name}_aggregated_results.csv', index=False)
        create_plots(aggregated_df, dataset_name, output_dir)
    
    return aggregated_df

def create_plots(aggregated_df, dataset_name, output_dir):
    """Create plots for runtime vs branching cost and colless index vs branching cost."""
    if len(aggregated_df) == 0:
        print(f"  No data to plot for {dataset_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Runtime vs Branching Cost
    ax1.errorbar(
        aggregated_df['branching_cost'],
        aggregated_df['mean_total_time'],
        yerr=aggregated_df['std_total_time'],
        marker='o', capsize=5, capthick=2, linewidth=2
    )
    ax1.set_xlabel('Branching Cost', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title(f'Runtime vs Branching Cost\n{dataset_name}', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('linear')
    
    # Plot 2: Colless Index vs Branching Cost
    ax2.errorbar(
        aggregated_df['branching_cost'],
        aggregated_df['overall_mean_colless'],
        yerr=aggregated_df['overall_std_colless'],
        marker='o', capsize=5, capthick=2, linewidth=2, color='orange'
    )
    ax2.set_xlabel('Branching Cost', fontsize=12)
    ax2.set_ylabel('Normalized Colless Index', fontsize=12)
    ax2.set_title(f'Colless Index vs Branching Cost\n{dataset_name}', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('linear')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots to {output_dir / f'{dataset_name}_plots.png'}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Sweep over branching costs and compute R-sets with colless index analysis'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (without .csv extension)')
    parser.add_argument('--branching_cost', type=float, default=None,
                        help='Single branching cost to test (if not specified, sweeps over default range)')
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH,
                        help=f'Tree depth budget (default: {DEFAULT_DEPTH})')
    parser.add_argument('--lam', type=float, default=DEFAULT_LAM,
                        help=f'Regularization parameter (default: {DEFAULT_LAM})')
    parser.add_argument('--eps', type=float, default=DEFAULT_EPS,
                        help=f'Epsilon for R-set budget (default: {DEFAULT_EPS})')
    parser.add_argument('--min_support', type=float, default=DEFAULT_MIN_SUPPORT,
                        help=f'Minimum support for splits (default: {DEFAULT_MIN_SUPPORT})')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: branching_cost_sweep_results)')
    
    return parser.parse_args()

def main():
    """Main function to run experiment for a single dataset."""
    args = parse_arguments()
    
    # Create output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent / 'branching_cost_sweep_results'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find dataset file
    dataset_path = DATA_DIR / f'{args.dataset}.csv'
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    # Determine branching costs to test
    if args.branching_cost is not None:
        branching_costs = [args.branching_cost]
    else:
        branching_costs = DEFAULT_BRANCHING_COSTS
    
    # Run experiment
    try:
        aggregated_df = run_experiment(
            dataset_path, output_dir, branching_costs,
            args.depth, args.lam, args.eps, args.min_support
        )
        
        if aggregated_df is not None and len(aggregated_df) > 0:
            print(f"\n{'='*80}")
            print(f"Experiment completed successfully!")
            print(f"Results saved to {output_dir}")
            print(f"{'='*80}")
        else:
            print(f"\nWarning: No results generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

