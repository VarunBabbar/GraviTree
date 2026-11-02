"""
Aggregate results from falling_trees_vs_regular_trees array jobs and create plots.
Run this after all array jobs complete.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_with_optional_se(ax, x, y, y_se, label, marker, color):
    if y_se is not None and not np.all(np.isnan(y_se)):
        ax.errorbar(
            x,
            y,
            yerr=y_se,
            marker=marker,
            linewidth=2,
            markersize=8,
            label=label,
            capsize=5,
            capthick=2,
            color=color,
        )
    else:
        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=2,
            markersize=8,
            label=label,
            color=color,
        )


def create_plots(summary_df: pd.DataFrame, dataset_name: str, output_dir: Path):
    if summary_df is None or summary_df.empty:
        print(f"No summary data available for plotting: {dataset_name}")
        return

    summary_df = summary_df.sort_values("branching_cost")

    # Plot expected decision sparsity vs branching cost
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    _plot_with_optional_se(
        ax,
        summary_df["branching_cost"],
        summary_df["with_constraint_sparsity_mean"],
        summary_df.get("with_constraint_sparsity_se"),
        "Falling constraint ON",
        "o",
        "blue",
    )
    _plot_with_optional_se(
        ax,
        summary_df["branching_cost"],
        summary_df["without_constraint_sparsity_mean"],
        summary_df.get("without_constraint_sparsity_se"),
        "Falling constraint OFF",
        "^",
        "orange",
    )
    ax.set_xlabel("Branching Cost", fontsize=12)
    ax.set_ylabel("Expected Decision Sparsity (mean ± SE)", fontsize=12)
    ax.set_title(
        f"Expected Decision Sparsity vs Branching Cost\n{dataset_name}", fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sparsity_plot_path = output_dir / f"{dataset_name}_sparsity_vs_branching_cost.png"
    plt.savefig(sparsity_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sparsity plot to {sparsity_plot_path}")

    # Plot expected decision sparsity (positive class) vs branching cost
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    _plot_with_optional_se(
        ax,
        summary_df["branching_cost"],
        summary_df["with_constraint_sparsity_pos_mean"],
        summary_df.get("with_constraint_sparsity_pos_se"),
        "Falling constraint ON",
        "o",
        "blue",
    )
    _plot_with_optional_se(
        ax,
        summary_df["branching_cost"],
        summary_df["without_constraint_sparsity_pos_mean"],
        summary_df.get("without_constraint_sparsity_pos_se"),
        "Falling constraint OFF",
        "^",
        "orange",
    )
    ax.set_xlabel("Branching Cost", fontsize=12)
    ax.set_ylabel("Expected Decision Sparsity (pos, mean ± SE)", fontsize=12)
    ax.set_title(
        f"Decision Sparsity (Positive Class) vs Branching Cost\n{dataset_name}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pos_plot_path = output_dir / f"{dataset_name}_sparsity_pos_vs_branching_cost.png"
    plt.savefig(pos_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved positive-class sparsity plot to {pos_plot_path}")

    # Plot expected decision sparsity (negative class) vs branching cost
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    _plot_with_optional_se(
        ax,
        summary_df["branching_cost"],
        summary_df["with_constraint_sparsity_neg_mean"],
        summary_df.get("with_constraint_sparsity_neg_se"),
        "Falling constraint ON",
        "o",
        "blue",
    )
    _plot_with_optional_se(
        ax,
        summary_df["branching_cost"],
        summary_df["without_constraint_sparsity_neg_mean"],
        summary_df.get("without_constraint_sparsity_neg_se"),
        "Falling constraint OFF",
        "^",
        "orange",
    )
    ax.set_xlabel("Branching Cost", fontsize=12)
    ax.set_ylabel("Expected Decision Sparsity (neg, mean ± SE)", fontsize=12)
    ax.set_title(
        f"Decision Sparsity (Negative Class) vs Branching Cost\n{dataset_name}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    neg_plot_path = output_dir / f"{dataset_name}_sparsity_neg_vs_branching_cost.png"
    plt.savefig(neg_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved negative-class sparsity plot to {neg_plot_path}")

    # Plot Rashomon set size vs branching cost (log scale)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    with_rset = summary_df["with_constraint_rset_size_mean"].clip(lower=0.1)
    without_rset = summary_df["without_constraint_rset_size_mean"].clip(lower=0.1)
    ax.plot(
        summary_df["branching_cost"],
        with_rset,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Falling constraint ON",
        color="blue",
    )
    ax.plot(
        summary_df["branching_cost"],
        without_rset,
        marker="^",
        linewidth=2,
        markersize=8,
        label="Falling constraint OFF",
        color="orange",
    )
    ax.set_xlabel("Branching Cost", fontsize=12)
    ax.set_ylabel("Rashomon Set Size (mean)", fontsize=12)
    ax.set_title(f"Rashomon Set Size vs Branching Cost\n{dataset_name}", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    rset_plot_path = output_dir / f"{dataset_name}_rset_size_vs_branching_cost.png"
    plt.savefig(rset_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Rashomon set size plot to {rset_plot_path}")

    # Plot runtime vs branching cost (means only)
    if "with_constraint_time_mean" in summary_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(
            summary_df["branching_cost"],
            summary_df["with_constraint_time_mean"],
            marker="o",
            linewidth=2,
            markersize=8,
            label="Falling constraint ON",
            color="blue",
        )
        ax.plot(
            summary_df["branching_cost"],
            summary_df["without_constraint_time_mean"],
            marker="^",
            linewidth=2,
            markersize=8,
            label="Falling constraint OFF",
            color="orange",
        )
        ax.set_xlabel("Branching Cost", fontsize=12)
        ax.set_ylabel("Runtime (mean seconds)", fontsize=12)
        ax.set_title(f"Runtime vs Branching Cost\n{dataset_name}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        runtime_plot_path = output_dir / f"{dataset_name}_runtime_vs_branching_cost.png"
        plt.savefig(runtime_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved runtime plot to {runtime_plot_path}")


def create_loss_vs_sparsity_scatterplot(detailed_df: pd.DataFrame, dataset_name: str, output_dir: Path):
    if detailed_df is None or detailed_df.empty:
        print(f"No detailed data available for scatterplot: {dataset_name}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(
        detailed_df["with_constraint_sparsity_mean"],
        detailed_df["with_constraint_best_loss"],
        color="blue",
        alpha=0.6,
        s=50,
        marker="o",
        label="Falling constraint ON",
    )
    ax.scatter(
        detailed_df["without_constraint_sparsity_mean"],
        detailed_df["without_constraint_best_loss"],
        color="orange",
        alpha=0.6,
        s=50,
        marker="^",
        label="Falling constraint OFF",
    )
    ax.set_xlabel("Expected Decision Sparsity (mean)", fontsize=12)
    ax.set_ylabel("Best Loss", fontsize=12)
    ax.set_title(f"Loss vs Sparsity\n{dataset_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"{dataset_name}_loss_vs_sparsity_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss vs sparsity scatterplot to {out_path}")


def create_loss_vs_sparsity_scatterplot_by_class(
    detailed_df: pd.DataFrame, dataset_name: str, output_dir: Path
):
    if detailed_df is None or detailed_df.empty:
        print(f"No detailed data available for class scatterplots: {dataset_name}")
        return

    for class_label, col_suffix in [("Positive", "pos"), ("Negative", "neg")]:
        with_sparsity_col = f"with_constraint_sparsity_{col_suffix}_mean"
        without_sparsity_col = f"without_constraint_sparsity_{col_suffix}_mean"
        with_loss_col = f"with_constraint_loss_{col_suffix}_mean"
        without_loss_col = f"without_constraint_loss_{col_suffix}_mean"
        required_cols = {
            with_sparsity_col,
            without_sparsity_col,
            with_loss_col,
            without_loss_col,
        }
        if not required_cols.issubset(detailed_df.columns):
            print(f"Missing class loss/sparsity columns for {class_label}: {dataset_name}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.scatter(
            detailed_df[with_sparsity_col],
            detailed_df[with_loss_col],
            color="blue",
            alpha=0.6,
            s=50,
            marker="o",
            label="Falling constraint ON",
        )
        ax.scatter(
            detailed_df[without_sparsity_col],
            detailed_df[without_loss_col],
            color="orange",
            alpha=0.6,
            s=50,
            marker="^",
            label="Falling constraint OFF",
        )
        ax.set_xlabel(f"Expected Decision Sparsity ({class_label.lower()}, mean)", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Loss vs Sparsity ({class_label} Class)\n{dataset_name}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = output_dir / f"{dataset_name}_loss_vs_sparsity_{col_suffix}_scatter.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {class_label.lower()}-class loss vs sparsity scatterplot to {out_path}")


def aggregate_results(results_dir: str):
    """Aggregate summary CSVs from array jobs and create plots."""
    results_dir = Path(results_dir)

    pattern_csv = str(results_dir / "*_bc_*_summary.csv")
    csv_files = glob.glob(pattern_csv)

    if len(csv_files) == 0:
        print(f"No summary CSV files found in {results_dir}")
        return None

    dataset_results = {}
    for csv_file in csv_files:
        filename = Path(csv_file).stem
        parts = filename.split("_bc_")
        if len(parts) != 2:
            continue
        dataset_name = parts[0]
        bc_str = parts[1].replace("_summary", "")
        try:
            bc = float(bc_str.replace("_", "."))
        except ValueError:
            continue

        df = pd.read_csv(csv_file)
        df["dataset"] = dataset_name
        dataset_results.setdefault(dataset_name, []).append(df)

    for dataset_name, dfs in dataset_results.items():
        if len(dfs) == 0:
            continue
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values("branching_cost")

        # Replace NaN SE values with 0 so error bars don't break
        for col in combined_df.columns:
            if col.endswith("_se"):
                combined_df[col] = combined_df[col].fillna(0.0)

        # Load detailed results for scatterplot
        detailed_files = glob.glob(str(results_dir / f"{dataset_name}_bc_*_detailed_results.csv"))
        if len(detailed_files) == 0:
            detailed_files = glob.glob(str(results_dir / f"{dataset_name}_full_detailed_results.csv"))
        detailed_df = None
        if len(detailed_files) > 0:
            detailed_frames = []
            for file in detailed_files:
                try:
                    detailed_frames.append(pd.read_csv(file))
                except Exception:
                    continue
            if len(detailed_frames) > 0:
                detailed_df = pd.concat(detailed_frames, ignore_index=True)

        create_plots(combined_df, dataset_name, results_dir)
        if detailed_df is not None:
            create_loss_vs_sparsity_scatterplot(detailed_df, dataset_name, results_dir)
            create_loss_vs_sparsity_scatterplot_by_class(detailed_df, dataset_name, results_dir)
        combined_df.to_csv(
            results_dir / f"{dataset_name}_aggregated_summary.csv", index=False
        )
        print(f"Processed dataset: {dataset_name} ({len(combined_df)} branching costs)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate falling_trees_vs_regular_trees results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="falling_trees_vs_regular_trees_results",
        help="Directory containing results",
    )

    args = parser.parse_args()
    aggregate_results(args.results_dir)
    print("\nAggregation complete!")

