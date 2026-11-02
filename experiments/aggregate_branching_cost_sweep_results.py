"""
Post-processing script to aggregate results from array jobs and create plots per dataset.
Run this after all array jobs complete to combine results and generate plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def aggregate_results_for_dataset(results_dir, dataset_name):
    """Aggregate all results for a single dataset across all branching costs."""
    # Find all detailed results files for this dataset
    # Pattern 1: Full sweep results (dataset_detailed_results.csv)
    pattern1 = str(results_dir / f'{dataset_name}_detailed_results.csv')
    # Pattern 2: Array job results (dataset_bc_*_detailed_results.csv)
    pattern2 = str(results_dir / f'{dataset_name}_bc_*_detailed_results.csv')
    
    files = glob.glob(pattern1) + glob.glob(pattern2)
    
    if len(files) == 0:
        print(f"No results found for dataset: {dataset_name}")
        return None
    
    # Load all results
    all_results = []
    for file in files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if len(all_results) == 0:
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Aggregate by branching_cost across all splits
    aggregated = []
    for bc in sorted(combined_df['branching_cost'].unique()):
        bc_results = combined_df[combined_df['branching_cost'] == bc]
        if len(bc_results) == 0:
            continue
        
        # Aggregate metrics
        agg_result = {
            'branching_cost': bc,
            'dataset': dataset_name,
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
        
        # Aggregate all colless indices across all splits
        all_colless = []
        for colless_list in bc_results['colless_indices']:
            if isinstance(colless_list, str):
                # Parse string representation of list
                import ast
                colless_list = ast.literal_eval(colless_list)
            all_colless.extend(colless_list)
        
        if all_colless:
            agg_result['overall_mean_colless'] = np.mean(all_colless)
            agg_result['overall_std_colless'] = np.std(all_colless)
        else:
            agg_result['overall_mean_colless'] = 0.0
            agg_result['overall_std_colless'] = 0.0
        
        aggregated.append(agg_result)
    
    aggregated_df = pd.DataFrame(aggregated)
    return aggregated_df

def create_plots(aggregated_df, dataset_name, output_dir):
    """Create plots for runtime vs branching cost, colless index vs branching cost, and R-set size vs branching cost."""
    if len(aggregated_df) == 0:
        print(f"  No data to plot for {dataset_name}")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
    
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
    
    # Plot 3: Rashomon Set Size vs Branching Cost
    ax3.errorbar(
        aggregated_df['branching_cost'],
        aggregated_df['mean_rset_size'],
        yerr=aggregated_df['std_rset_size'],
        marker='o', capsize=5, capthick=2, linewidth=2, color='green'
    )
    ax3.set_xlabel('Branching Cost', fontsize=12)
    ax3.set_ylabel('Rashomon Set Size', fontsize=12)
    ax3.set_title(f'Rashomon Set Size vs Branching Cost\n{dataset_name}', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('linear')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots to {output_dir / f'{dataset_name}_plots.png'}")

def main():
    """Main function to aggregate results and create plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate results from array jobs and create plots')
    parser.add_argument('--results_dir', type=str, 
                       default='branching_cost_sweep_results',
                       help='Directory containing results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Get all unique datasets from detailed results files
    pattern = str(results_dir / '*_detailed_results.csv')
    files = glob.glob(pattern)
    
    datasets = set()
    for file in files:
        filename = Path(file).stem
        # Handle both patterns:
        # - dataset_detailed_results -> dataset
        # - dataset_bc_0_02_detailed_results -> dataset
        if filename.endswith('_detailed_results'):
            dataset_name = filename.replace('_detailed_results', '')
            # Remove _bc_* suffix if present
            if '_bc_' in dataset_name:
                dataset_name = dataset_name.split('_bc_')[0]
            datasets.add(dataset_name)
    
    print(f"Found {len(datasets)} datasets to aggregate")
    print(f"Datasets: {sorted(datasets)}")
    print()
    
    # Aggregate and plot for each dataset
    all_aggregated = []
    for dataset_name in sorted(datasets):
        print(f"Processing dataset: {dataset_name}")
        aggregated_df = aggregate_results_for_dataset(results_dir, dataset_name)
        
        if aggregated_df is not None and len(aggregated_df) > 0:
            # Save aggregated results
            aggregated_df.to_csv(results_dir / f'{dataset_name}_aggregated_results.csv', index=False)
            
            # Create plots
            create_plots(aggregated_df, dataset_name, results_dir)
            
            all_aggregated.append(aggregated_df)
            print(f"  ✓ Aggregated {len(aggregated_df)} branching cost configurations")
        else:
            print(f"  ✗ No valid results to aggregate")
        print()
    
    # Combine all datasets
    if len(all_aggregated) > 0:
        combined_df = pd.concat(all_aggregated, ignore_index=True)
        combined_df.to_csv(results_dir / 'all_datasets_combined_results.csv', index=False)
        print(f"Combined results saved to {results_dir / 'all_datasets_combined_results.csv'}")
    
    print("Aggregation complete!")

if __name__ == "__main__":
    main()

