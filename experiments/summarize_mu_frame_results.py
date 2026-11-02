"""
Summarize Falling Trees vs FRAME from each dataset's full detailed CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _load_full_results(dataset: str, results_dir: Path) -> pd.DataFrame:
    full_path = results_dir / f"{dataset}_full_detailed_results.csv"
    if full_path.exists():
        return pd.read_csv(full_path)
    return pd.DataFrame()


def _filter_bc(df: pd.DataFrame, mu: float) -> pd.DataFrame:
    if df.empty or "branching_cost" not in df.columns:
        return df.iloc[0:0]
    return df[np.isclose(df["branching_cost"].astype(float), mu)]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    return float(np.mean(values)) if values else 0.0, float(np.std(values)) if values else 0.0


def _extract_metric(df: pd.DataFrame, metric: str, rset_col: str) -> Tuple[float, float]:
    if df.empty or metric not in df.columns:
        return 0.0, 0.0
    if rset_col in df.columns:
        df = df[df[rset_col] > 0]
    if df.empty:
        return 0.0, 0.0
    return _mean_std(df[metric].tolist())


def _mean_or_nan(values: List[float]) -> float:
    values = [v for v in values if pd.notna(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize_dataset(
    dataset: str,
    mu: float,
    results_dir: Path,
) -> Dict[str, float]:
    row = {"dataset": dataset}
    full_df = _load_full_results(dataset, results_dir)
    bc_df = _filter_bc(full_df, mu)
    if bc_df.empty:
        return row

    ft_rset_mean, ft_rset_std = _extract_metric(
        bc_df, "falling_trees_rset_size", "falling_trees_rset_size"
    )
    ft_sparsity_mean, ft_sparsity_std = _extract_metric(
        bc_df, "falling_trees_sparsity_pos_mean", "falling_trees_rset_size"
    )
    ft_loss_pos_mean, ft_loss_pos_std = _extract_metric(
        bc_df, "falling_trees_loss_pos_mean", "falling_trees_rset_size"
    )
    ft_loss_mean, ft_loss_std = _extract_metric(
        bc_df, "falling_trees_loss_mean", "falling_trees_rset_size"
    )

    frame_rset_mean, frame_rset_std = _extract_metric(
        bc_df, "frame_rset_size", "frame_rset_size"
    )
    frame_sparsity_mean, frame_sparsity_std = _extract_metric(
        bc_df, "frame_sparsity_pos_mean", "frame_rset_size"
    )
    frame_loss_pos_mean, frame_loss_pos_std = _extract_metric(
        bc_df, "frame_loss_pos_mean", "frame_rset_size"
    )
    frame_loss_mean, frame_loss_std = _extract_metric(
        bc_df, "frame_loss_mean", "frame_rset_size"
    )

    row[f"mu_{mu}_ft_rset_size_mean"] = ft_rset_mean
    row[f"mu_{mu}_ft_rset_size_std"] = ft_rset_std
    row[f"mu_{mu}_ft_sparsity_pos_mean"] = ft_sparsity_mean
    row[f"mu_{mu}_ft_sparsity_pos_std"] = ft_sparsity_std
    row[f"mu_{mu}_ft_loss_pos_mean"] = ft_loss_pos_mean
    row[f"mu_{mu}_ft_loss_pos_std"] = ft_loss_pos_std
    row[f"mu_{mu}_ft_loss_mean"] = ft_loss_mean
    row[f"mu_{mu}_ft_loss_std"] = ft_loss_std

    row[f"mu_{mu}_frame_rset_size_mean"] = frame_rset_mean
    row[f"mu_{mu}_frame_rset_size_std"] = frame_rset_std
    row[f"mu_{mu}_frame_sparsity_pos_mean"] = frame_sparsity_mean
    row[f"mu_{mu}_frame_sparsity_pos_std"] = frame_sparsity_std
    row[f"mu_{mu}_frame_loss_pos_mean"] = frame_loss_pos_mean
    row[f"mu_{mu}_frame_loss_pos_std"] = frame_loss_pos_std
    row[f"mu_{mu}_frame_loss_mean"] = frame_loss_mean
    row[f"mu_{mu}_frame_loss_std"] = frame_loss_std

    return row


def _collect_violin_data(
    datasets: List[str],
    mu_values: List[float],
    results_dir: Path,
    max_len: int,
) -> pd.DataFrame:
    metric_map = {
        "loss": ("falling_trees_loss_mean", "frame_loss_mean"),
        "pos_loss": ("falling_trees_loss_pos_mean", "frame_loss_pos_mean"),
        "rset_size": ("falling_trees_rset_size", "frame_rset_size"),
        "pos_sparsity": ("falling_trees_sparsity_pos_mean", "frame_sparsity_pos_mean"),
    }
    records = []
    for dataset in datasets:
        full_df = _load_full_results(dataset, results_dir)
        if full_df.empty:
            continue
        for mu in mu_values:
            bc_df = _filter_bc(full_df, mu)
            if bc_df.empty:
                continue
            for metric_key, (ft_col, frame_col) in metric_map.items():
                for label, col_name in (
                    ("GraviTree", ft_col),
                    ("FRAME", frame_col),
                ):
                    values = bc_df.get(col_name, pd.Series(dtype=float)).tolist()
                    mean_value = _mean_or_nan(values)
                    if np.isnan(mean_value):
                        continue
                    records.append(
                        {
                            "dataset": dataset,
                            "mu": f"{mu:g}",
                            "metric": metric_key,
                            "method": label,
                            "value": mean_value,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _plot_violin_summary(df: pd.DataFrame, mu_values: List[float], max_len: int, output_path: Path) -> None:
    if df.empty:
        print("No data available for violin plots.")
        return
    metric_titles = [
        ("loss", "Loss"),
        ("pos_loss", "Pos Loss"),
        ("rset_size", "Rashomon Set Size"),
        ("pos_sparsity", "Pos Sparsity (+)"),
    ]
    for mu in mu_values:
        mu_key = f"{mu:g}"
        mu_df = df[df["mu"] == mu_key]
        if mu_df.empty:
            print(f"No data available for mu={mu_key}.")
            continue
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        for ax, (metric_key, title) in zip(axes, metric_titles):
            metric_df = mu_df[mu_df["metric"] == metric_key]
            if metric_df.empty:
                ax.set_title(f"{title} (no data)")
                ax.axis("off")
                continue
            sns.violinplot(
                data=metric_df,
                x="method",
                y="value",
                ax=ax,
                inner="box",
                cut=0,
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if metric_key == "rset_size":
                ax.set_yscale("log")
        fig.suptitle(f"GraviTree (μ = {mu_key}) vs FRAME (max len = {max_len})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        mu_output = output_path.with_name(f"{output_path.stem}_mu_{mu_key}_max_len_{max_len}{output_path.suffix}")
        fig.savefig(mu_output, dpi=300)
        print(f"Saved violin plot to {mu_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Falling Trees vs FRAME from full detailed CSVs."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets (e.g., compas,bar,bar7)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.1,
        help="Branching cost value to summarize (default: 0.1)",
    )
    parser.add_argument(
        "--mu_values",
        type=str,
        default=None,
        help="Comma-separated mu values to plot (overrides --mu)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="falling_trees_vs_frame_runtime_results_max_len_1",
        help="Directory containing falling_trees_vs_frame_runtime results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mu_summary_frame.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default="mu_summary_frame_violin.png",
        help="Output filename for the violin plot summary",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1,
        help="Maximum length of the trees to summarize (default: 1)",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    results_dir = Path(args.results_dir)
    if args.mu_values:
        mu_values = [float(x.strip()) for x in args.mu_values.split(",") if x.strip()]
    else:
        mu_values = [args.mu]

    rows = []
    for dataset in datasets:
        for mu in mu_values:
            rows.append(summarize_dataset(dataset, mu, results_dir))

    out_path = Path(args.output)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")

    violin_df = _collect_violin_data(datasets, mu_values, results_dir, args.max_len)
    _plot_violin_summary(violin_df, mu_values, args.max_len, Path(args.plot_output))


if __name__ == "__main__":
    main()

