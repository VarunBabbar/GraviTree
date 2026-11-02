"""
Aggregate results from falling_trees_vs_frame_runtime array jobs and create plots.
Run this after all array jobs complete.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import pickle

def aggregate_results(results_dir):
    """Aggregate results from all array job outputs."""
    results_dir = Path(results_dir)
    
    # Load from summary CSV files
    pattern_csv = str(results_dir / '*_bc_*_summary.csv')
    # This finds all summary CSV files (with names like dataset_bc_*_summary.csv) in the specified results directory.
    csv_files = glob.glob(pattern_csv)
    
    if len(csv_files) == 0:
        print(f"No summary CSV files found in {results_dir}")
        return None
    
    # Parse dataset and branching cost from filename
    dataset_results = {}
    for csv_file in csv_files:
        filename = Path(csv_file).stem
        # Format: dataset_bc_0_02_summary
        parts = filename.split('_bc_')
        if len(parts) == 2:
            dataset_name = parts[0]
            bc_str = parts[1].replace('_summary', '')
            bc = float(bc_str.replace('_', '.'))
            
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = []
            
            df = pd.read_csv(csv_file)
            df['dataset'] = dataset_name
            dataset_results[dataset_name].append(df)
    
    # Aggregate and plot for each dataset
    for dataset_name, dfs in dataset_results.items():
        if len(dfs) == 0:
            continue
        
        # Combine all branching costs for this dataset
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('branching_cost')
        
        # Load detailed results for scatterplots
        # Collect all results for each branching cost (across all splits)
        detailed_results = []
        for bc in combined_df['branching_cost']:
            bc_str = str(bc).replace('.', '_')
            # Try to find results files (could be with or without split index)
            pkl_files_for_bc = []
            # Pattern 1: dataset_split_*_bc_*_results.pkl (with split index)
            pattern_with_split = str(results_dir / f'{dataset_name}_split_0_bc_{bc_str}_results.pkl')
            pkl_files_for_bc.extend(glob.glob(pattern_with_split))
            # Pattern 2: dataset_bc_*_results.pkl (without split index, for backward compatibility)
            pkl_file_no_split = results_dir / f'{dataset_name}_bc_{bc_str}_results.pkl'
            if pkl_file_no_split.exists():
                pkl_files_for_bc.append(str(pkl_file_no_split))
            
            # Load all results for this branching cost
            bc_results = []
            for pkl_file in pkl_files_for_bc:
                with open(pkl_file, 'rb') as f:
                    bc_results.append(pickle.load(f))
            
            if len(bc_results) > 0:
                # Aggregate results across splits
                aggregated_result = {
                    'branching_cost': bc,
                    'terms_falling_trees': [],
                    'loss_falling_trees_train': [],
                    'loss_falling_trees_test': [],
                    'terms_frame': [],
                    'loss_frame_train': [],
                    'loss_frame_test': [],
                    # ROC curve data (per model, on test set)
                    'roc_falling_trees': [],
                    'roc_frame': [],
                    # Class-based metrics
                    'terms_falling_trees_train_pos': [],
                    'terms_falling_trees_train_neg': [],
                    'loss_falling_trees_train_pos': [],
                    'loss_falling_trees_train_neg': [],
                    'terms_falling_trees_test_pos': [],
                    'terms_falling_trees_test_neg': [],
                    'loss_falling_trees_test_pos': [],
                    'loss_falling_trees_test_neg': [],
                    'terms_frame_train_pos': [],
                    'terms_frame_train_neg': [],
                    'loss_frame_train_pos': [],
                    'loss_frame_train_neg': [],
                    'terms_frame_test_pos': [],
                    'terms_frame_test_neg': [],
                    'loss_frame_test_pos': [],
                    'loss_frame_test_neg': [],
                }
                for r in bc_results:
                    aggregated_result['terms_falling_trees'].extend(r.get('terms_falling_trees', []))
                    aggregated_result['loss_falling_trees_train'].extend(r.get('loss_falling_trees_train', r.get('loss_falling_trees', [])))
                    aggregated_result['loss_falling_trees_test'].extend(r.get('loss_falling_trees_test', []))
                    aggregated_result['terms_frame'].extend(r.get('terms_frame', []))
                    aggregated_result['loss_frame_train'].extend(r.get('loss_frame_train', r.get('loss_frame', [])))
                    aggregated_result['loss_frame_test'].extend(r.get('loss_frame_test', []))
                    # ROC curves
                    aggregated_result['roc_falling_trees'].extend(r.get('roc_falling_trees', []))
                    aggregated_result['roc_frame'].extend(r.get('roc_frame', []))
                    # Class-based metrics
                    aggregated_result['terms_falling_trees_train_pos'].extend(r.get('terms_falling_trees_train_pos', []))
                    aggregated_result['terms_falling_trees_train_neg'].extend(r.get('terms_falling_trees_train_neg', []))
                    aggregated_result['loss_falling_trees_train_pos'].extend(r.get('loss_falling_trees_train_pos', []))
                    aggregated_result['loss_falling_trees_train_neg'].extend(r.get('loss_falling_trees_train_neg', []))
                    aggregated_result['terms_falling_trees_test_pos'].extend(r.get('terms_falling_trees_test_pos', []))
                    aggregated_result['terms_falling_trees_test_neg'].extend(r.get('terms_falling_trees_test_neg', []))
                    aggregated_result['loss_falling_trees_test_pos'].extend(r.get('loss_falling_trees_test_pos', []))
                    aggregated_result['loss_falling_trees_test_neg'].extend(r.get('loss_falling_trees_test_neg', []))
                    aggregated_result['terms_frame_train_pos'].extend(r.get('terms_frame_train_pos', []))
                    aggregated_result['terms_frame_train_neg'].extend(r.get('terms_frame_train_neg', []))
                    aggregated_result['loss_frame_train_pos'].extend(r.get('loss_frame_train_pos', []))
                    aggregated_result['loss_frame_train_neg'].extend(r.get('loss_frame_train_neg', []))
                    aggregated_result['terms_frame_test_pos'].extend(r.get('terms_frame_test_pos', []))
                    aggregated_result['terms_frame_test_neg'].extend(r.get('terms_frame_test_neg', []))
                    aggregated_result['loss_frame_test_pos'].extend(r.get('loss_frame_test_pos', []))
                    aggregated_result['loss_frame_test_neg'].extend(r.get('loss_frame_test_neg', []))
                
                detailed_results.append(aggregated_result)
        
        # Create plots
        create_runtime_plot(combined_df, dataset_name, results_dir)
        create_loss_plots(combined_df, dataset_name, results_dir)
        create_sparsity_plot(combined_df, dataset_name, results_dir)
        create_scatterplots(detailed_results, dataset_name, results_dir)
        create_class_based_plots(detailed_results, dataset_name, results_dir)
        create_roc_curve_plot(detailed_results, dataset_name, results_dir)
        
        # Save aggregated summary
        combined_df.to_csv(results_dir / f'{dataset_name}_aggregated_summary.csv', index=False)
        
        print(f"Processed dataset: {dataset_name} ({len(combined_df)} branching costs)")


def create_runtime_plot(summary_df, dataset_name, output_dir):
    """Create plot showing runtime vs branching cost with error bars."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Check if we have mean/se columns or just single values
    # Use standard error (se) if available, otherwise fall back to std or no error bars
    if 'falling_trees_time_mean' in summary_df.columns:
        # Plot Falling Trees runtime with error bars (standard error)
        time_se_col = 'falling_trees_time_se' if 'falling_trees_time_se' in summary_df.columns else 'falling_trees_time_std'
        if time_se_col in summary_df.columns:
            ax.errorbar(
                summary_df['branching_cost'],
                summary_df['falling_trees_time_mean'],
                yerr=summary_df[time_se_col],
                marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue',
                capsize=5, capthick=2
            )
        else:
            ax.plot(
                summary_df['branching_cost'],
                summary_df['falling_trees_time_mean'],
                marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue'
            )
        
        # Plot FRAME runtime (constant line, show mean±se in label)
        if 'frame_time_mean' in summary_df.columns:
            frame_time_mean = summary_df['frame_time_mean'].mean()
            frame_time_se_col = 'frame_time_se' if 'frame_time_se' in summary_df.columns else 'frame_time_std'
            if frame_time_se_col in summary_df.columns:
                frame_time_se = summary_df[frame_time_se_col].mean()
                ax.axhline(
                    y=frame_time_mean,
                    color='red', linestyle='--', linewidth=2,
                    label=f'FRAME (mean: {frame_time_mean:.2f}±{frame_time_se:.2f}s)'
                )
                # Add shaded region for standard error
                ax.axhspan(
                    frame_time_mean - frame_time_se,
                    frame_time_mean + frame_time_se,
                    alpha=0.2, color='red'
                )
            else:
                ax.axhline(
                    y=frame_time_mean,
                    color='red', linestyle='--', linewidth=2,
                    label=f'FRAME (mean: {frame_time_mean:.2f}s)'
                )
        else:
            frame_time_mean = summary_df['frame_time'].mean()
            ax.axhline(
                y=frame_time_mean,
                color='red', linestyle='--', linewidth=2, 
                label=f'FRAME (mean: {frame_time_mean:.2f}s)'
            )
    else:
        # Fallback for old format
        ax.plot(
            summary_df['branching_cost'],
            summary_df['falling_trees_time'],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue'
        )
        
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


def create_loss_plots(summary_df, dataset_name, output_dir):
    """Create plots showing train and test loss vs branching cost with error bars (standard error)."""
    # Check if we have the required columns
    if 'falling_trees_loss_train_mean' not in summary_df.columns:
        print(f"  Skipping loss plots - loss columns not found")
        return
    
    # Train loss plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use standard error (se) if available, otherwise fall back to std
    train_se_col_ft = 'falling_trees_loss_train_se' if 'falling_trees_loss_train_se' in summary_df.columns else 'falling_trees_loss_train_std'
    train_se_col_frame = 'frame_loss_train_se' if 'frame_loss_train_se' in summary_df.columns else 'frame_loss_train_std'
    
    if train_se_col_ft in summary_df.columns:
        ax.errorbar(
            summary_df['branching_cost'],
            summary_df['falling_trees_loss_train_mean'],
            yerr=summary_df[train_se_col_ft],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue',
            capsize=5, capthick=2
        )
        ax.errorbar(
            summary_df['branching_cost'],
            summary_df['frame_loss_train_mean'],
            yerr=summary_df[train_se_col_frame],
            marker='^', linewidth=2, markersize=8, label='FRAME', color='red',
            capsize=5, capthick=2
        )
    else:
        ax.plot(
            summary_df['branching_cost'],
            summary_df['falling_trees_loss_train_mean'],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue'
        )
        ax.plot(
            summary_df['branching_cost'],
            summary_df['frame_loss_train_mean'],
            marker='^', linewidth=2, markersize=8, label='FRAME', color='red'
        )
    
    ax.set_xlabel('Branching Cost', fontsize=12)
    ax.set_ylabel('Train Loss (mean ± SE)', fontsize=12)
    ax.set_title(f'Train Loss vs Branching Cost\n{dataset_name}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved train loss plot to {output_dir / f'{dataset_name}_train_loss.png'}")
    
    # Test loss plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    test_se_col_ft = 'falling_trees_loss_test_se' if 'falling_trees_loss_test_se' in summary_df.columns else 'falling_trees_loss_test_std'
    test_se_col_frame = 'frame_loss_test_se' if 'frame_loss_test_se' in summary_df.columns else 'frame_loss_test_std'
    
    if test_se_col_ft in summary_df.columns:
        ax.errorbar(
            summary_df['branching_cost'],
            summary_df['falling_trees_loss_test_mean'],
            yerr=summary_df[test_se_col_ft],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue',
            capsize=5, capthick=2
        )
        ax.errorbar(
            summary_df['branching_cost'],
            summary_df['frame_loss_test_mean'],
            yerr=summary_df[test_se_col_frame],
            marker='^', linewidth=2, markersize=8, label='FRAME', color='red',
            capsize=5, capthick=2
        )
    else:
        ax.plot(
            summary_df['branching_cost'],
            summary_df['falling_trees_loss_test_mean'],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue'
        )
        ax.plot(
            summary_df['branching_cost'],
            summary_df['frame_loss_test_mean'],
            marker='^', linewidth=2, markersize=8, label='FRAME', color='red'
        )
    
    ax.set_xlabel('Branching Cost', fontsize=12)
    ax.set_ylabel('Test Loss (mean ± SE)', fontsize=12)
    ax.set_title(f'Test Loss vs Branching Cost\n{dataset_name}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_test_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved test loss plot to {output_dir / f'{dataset_name}_test_loss.png'}")


def create_sparsity_plot(summary_df, dataset_name, output_dir):
    """Create plot showing decision sparsity vs branching cost with error bars (standard error)."""
    # Check if we have the required columns
    if 'falling_trees_sparsity_mean' not in summary_df.columns:
        print(f"  Skipping sparsity plot - sparsity columns not found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use standard error (se) if available, otherwise fall back to std
    sparsity_se_col_ft = 'falling_trees_sparsity_se' if 'falling_trees_sparsity_se' in summary_df.columns else 'falling_trees_sparsity_std'
    sparsity_se_col_frame = 'frame_sparsity_se' if 'frame_sparsity_se' in summary_df.columns else 'frame_sparsity_std'
    
    if sparsity_se_col_ft in summary_df.columns:
        ax.errorbar(
            summary_df['branching_cost'],
            summary_df['falling_trees_sparsity_mean'],
            yerr=summary_df[sparsity_se_col_ft],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue',
            capsize=5, capthick=2
        )
        ax.errorbar(
            summary_df['branching_cost'],
            summary_df['frame_sparsity_mean'],
            yerr=summary_df[sparsity_se_col_frame],
            marker='^', linewidth=2, markersize=8, label='FRAME', color='red',
            capsize=5, capthick=2
        )
    else:
        ax.plot(
            summary_df['branching_cost'],
            summary_df['falling_trees_sparsity_mean'],
            marker='o', linewidth=2, markersize=8, label='Falling Trees', color='blue'
        )
        ax.plot(
            summary_df['branching_cost'],
            summary_df['frame_sparsity_mean'],
            marker='^', linewidth=2, markersize=8, label='FRAME', color='red'
        )
    
    ax.set_xlabel('Branching Cost', fontsize=12)
    ax.set_ylabel('Decision Sparsity (mean ± SE)', fontsize=12)
    ax.set_title(f'Decision Sparsity vs Branching Cost\n{dataset_name}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved sparsity plot to {output_dir / f'{dataset_name}_sparsity.png'}")


def create_scatterplots(detailed_results, dataset_name, output_dir):
    """Create scatterplots of loss vs decision sparsity for each branching cost."""
    if len(detailed_results) == 0:
        return
    
    n_bc = len(detailed_results)
    
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
    
    # Create plots for TRAIN loss
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(detailed_results):
        ax = axes[idx]
        bc = result['branching_cost']
        
        # Scatter plot: train loss vs decision sparsity
        if len(result.get('terms_falling_trees', [])) > 0 and len(result.get('loss_falling_trees_train', [])) > 0:
            ax.scatter(
                result['terms_falling_trees'],
                result['loss_falling_trees_train'],
                color='blue', label='Falling Trees', alpha=0.6, s=50
            )
        if len(result.get('terms_frame', [])) > 0 and len(result.get('loss_frame_train', [])) > 0:
            ax.scatter(
                result['terms_frame'],
                result['loss_frame_train'],
                color='red', label='FRAME', alpha=0.6, s=50, marker='^'
            )
        
        ax.set_xlabel('Decision Sparsity', fontsize=10)
        ax.set_ylabel('Train Loss', fontsize=10)
        ax.set_title(f'Train Loss vs Decision Sparsity\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_train_loss_vs_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved train scatterplots to {output_dir / f'{dataset_name}_train_loss_vs_sparsity.png'}")
    
    # Create plots for TEST loss
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(detailed_results):
        ax = axes[idx]
        bc = result['branching_cost']
        
        # Scatter plot: test loss vs decision sparsity
        if len(result.get('terms_falling_trees', [])) > 0 and len(result.get('loss_falling_trees_test', [])) > 0:
            ax.scatter(
                result['terms_falling_trees'],
                result['loss_falling_trees_test'],
                color='blue', label='Falling Trees', alpha=0.6, s=50
            )
        if len(result.get('terms_frame', [])) > 0 and len(result.get('loss_frame_test', [])) > 0:
            ax.scatter(
                result['terms_frame'],
                result['loss_frame_test'],
                color='red', label='FRAME', alpha=0.6, s=50, marker='^'
            )
        
        ax.set_xlabel('Decision Sparsity', fontsize=10)
        ax.set_ylabel('Test Loss', fontsize=10)
        ax.set_title(f'Test Loss vs Decision Sparsity\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_test_loss_vs_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved test scatterplots to {output_dir / f'{dataset_name}_test_loss_vs_sparsity.png'}")


def create_class_based_plots(detailed_results, dataset_name, output_dir):
    """Create plots showing decision sparsity vs loss separately for positive and negative examples.
    
    Creates 4 PNG files:
    - train_positive_class_based_sparsity_vs_loss.png: grid of subplots, one per branching cost
    - train_negative_class_based_sparsity_vs_loss.png: grid of subplots, one per branching cost
    - test_positive_class_based_sparsity_vs_loss.png: grid of subplots, one per branching cost
    - test_negative_class_based_sparsity_vs_loss.png: grid of subplots, one per branching cost
    """
    if len(detailed_results) == 0:
        return
    
    n_bc = len(detailed_results)
    
    # Determine grid layout (same as create_scatterplots)
    if n_bc <= 3:
        n_cols = n_bc
        n_rows = 1
    elif n_bc <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_bc + 2) // 3
    
    # ========== TRAIN: Positive Examples ==========
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(detailed_results):
        ax = axes[idx]
        bc = result['branching_cost']
        
        # Falling Trees - positive
        if len(result.get('terms_falling_trees_train_pos', [])) > 0 and len(result.get('loss_falling_trees_train_pos', [])) > 0:
            ax.scatter(
                result['terms_falling_trees_train_pos'],
                result['loss_falling_trees_train_pos'],
                color='blue', label='Falling Trees', alpha=0.6, s=50, marker='o'
            )
        # FRAME - positive
        if len(result.get('terms_frame_train_pos', [])) > 0 and len(result.get('loss_frame_train_pos', [])) > 0:
            ax.scatter(
                result['terms_frame_train_pos'],
                result['loss_frame_train_pos'],
                color='red', label='FRAME', alpha=0.6, s=50, marker='^'
            )
        
        ax.set_xlabel('Decision Sparsity', fontsize=10)
        ax.set_ylabel('Loss on Positive Examples', fontsize=10)
        ax.set_title(f'Train: Positive Examples\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_train_positive_class_based_sparsity_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved train positive class plots to {output_dir / f'{dataset_name}_train_positive_class_based_sparsity_vs_loss.png'}")
    
    # ========== TRAIN: Negative Examples ==========
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(detailed_results):
        ax = axes[idx]
        bc = result['branching_cost']
        
        # Falling Trees - negative
        if len(result.get('terms_falling_trees_train_neg', [])) > 0 and len(result.get('loss_falling_trees_train_neg', [])) > 0:
            ax.scatter(
                result['terms_falling_trees_train_neg'],
                result['loss_falling_trees_train_neg'],
                color='blue', label='Falling Trees', alpha=0.6, s=50, marker='o'
            )
        # FRAME - negative
        if len(result.get('terms_frame_train_neg', [])) > 0 and len(result.get('loss_frame_train_neg', [])) > 0:
            ax.scatter(
                result['terms_frame_train_neg'],
                result['loss_frame_train_neg'],
                color='red', label='FRAME', alpha=0.6, s=50, marker='^'
            )
        
        ax.set_xlabel('Decision Sparsity', fontsize=10)
        ax.set_ylabel('Loss on Negative Examples', fontsize=10)
        ax.set_title(f'Train: Negative Examples\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_train_negative_class_based_sparsity_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved train negative class plots to {output_dir / f'{dataset_name}_train_negative_class_based_sparsity_vs_loss.png'}")
    
    # ========== TEST: Positive Examples ==========
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(detailed_results):
        ax = axes[idx]
        bc = result['branching_cost']
        
        # Falling Trees - positive
        if len(result.get('terms_falling_trees_test_pos', [])) > 0 and len(result.get('loss_falling_trees_test_pos', [])) > 0:
            ax.scatter(
                result['terms_falling_trees_test_pos'],
                result['loss_falling_trees_test_pos'],
                color='blue', label='Falling Trees', alpha=0.6, s=50, marker='o'
            )
        # FRAME - positive
        if len(result.get('terms_frame_test_pos', [])) > 0 and len(result.get('loss_frame_test_pos', [])) > 0:
            ax.scatter(
                result['terms_frame_test_pos'],
                result['loss_frame_test_pos'],
                color='red', label='FRAME', alpha=0.6, s=50, marker='^'
            )
        
        ax.set_xlabel('Decision Sparsity', fontsize=10)
        ax.set_ylabel('Loss on Positive Examples', fontsize=10)
        ax.set_title(f'Test: Positive Examples\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_test_positive_class_based_sparsity_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved test positive class plots to {output_dir / f'{dataset_name}_test_positive_class_based_sparsity_vs_loss.png'}")
    
    # ========== TEST: Negative Examples ==========
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_bc == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(detailed_results):
        ax = axes[idx]
        bc = result['branching_cost']
        
        # Falling Trees - negative
        if len(result.get('terms_falling_trees_test_neg', [])) > 0 and len(result.get('loss_falling_trees_test_neg', [])) > 0:
            ax.scatter(
                result['terms_falling_trees_test_neg'],
                result['loss_falling_trees_test_neg'],
                color='blue', label='Falling Trees', alpha=0.6, s=50, marker='o'
            )
        # FRAME - negative
        if len(result.get('terms_frame_test_neg', [])) > 0 and len(result.get('loss_frame_test_neg', [])) > 0:
            ax.scatter(
                result['terms_frame_test_neg'],
                result['loss_frame_test_neg'],
                color='red', label='FRAME', alpha=0.6, s=50, marker='^'
            )
        
        ax.set_xlabel('Decision Sparsity', fontsize=10)
        ax.set_ylabel('Loss on Negative Examples', fontsize=10)
        ax.set_title(f'Test: Negative Examples\nBranching Cost = {bc}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_bc, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_test_negative_class_based_sparsity_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved test negative class plots to {output_dir / f'{dataset_name}_test_negative_class_based_sparsity_vs_loss.png'}")


def create_roc_curve_plot(detailed_results, dataset_name, output_dir):
    """Create a single ROC curve plot overlaying all models in the Rashomon sets.
    
    Falling Trees curves are plotted in blue, FRAME curves in red, both with low alpha.
    """
    # Collect all ROC curves across branching costs and splits
    all_ft_rocs = []
    all_frame_rocs = []
    
    for result in detailed_results:
        all_ft_rocs.extend(result.get('roc_falling_trees', []))
        all_frame_rocs.extend(result.get('roc_frame', []))
    
    if len(all_ft_rocs) == 0 and len(all_frame_rocs) == 0:
        print(f"  Skipping ROC curve plot - no ROC data found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot Falling Trees ROC curves
    for roc in all_ft_rocs[:20]:
        fpr = np.array(roc['fpr'])
        tpr = np.array(roc['tpr'])
        ax.plot(fpr, tpr, color='blue', alpha=0.1)
    
    # Plot FRAME ROC curves
    for roc in all_frame_rocs[:20]:
        fpr = np.array(roc['fpr'])
        tpr = np.array(roc['tpr'])
        ax.plot(fpr, tpr, color='red', alpha=0.1)
    
    # Reference line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: Falling Trees vs FRAME Rashomon Sets\n{dataset_name}', fontsize=13)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Falling Trees (all models)'),
        Line2D([0], [0], color='red', lw=2, label='FRAME (all models)'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Random (y = x)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = output_dir / f'{dataset_name}_ROC_curve_falling_tree_vs_FRL.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved ROC curve plot to {out_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate falling_trees_vs_frame_runtime results')
    parser.add_argument('--results-dir', type=str, 
                       default='falling_trees_vs_frame_runtime_results_max_len_1',
                       help='Directory containing results')
    
    args = parser.parse_args()
    
    aggregate_results(args.results_dir)
    print("\nAggregation complete!")
