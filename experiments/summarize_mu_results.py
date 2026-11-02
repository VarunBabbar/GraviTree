"""
Summarize mu-specific metrics for Falling Trees vs Regular Trees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _mean_std(values: List[float]) -> Tuple[float, float]:
    values = [v for v in values if v > 0]
    if not values:
        return 0.0, 0.0
    return _mean(values), _std(values)


def _load_detailed_results(dataset: str, results_dir: Path) -> pd.DataFrame:
    full_path = results_dir / f"{dataset}_full_detailed_results.csv"
    if full_path.exists():
        return pd.read_csv(full_path)

    detailed_files = sorted(results_dir.glob(f"{dataset}_bc_*_detailed_results.csv"))
    if not detailed_files:
        return pd.DataFrame()

    return pd.concat([pd.read_csv(path) for path in detailed_files], ignore_index=True)


def _filter_bc(df: pd.DataFrame, mu: float) -> pd.DataFrame:
    if df.empty or "branching_cost" not in df.columns:
        return df.iloc[0:0]
    return df[np.isclose(df["branching_cost"].astype(float), mu)]


def _mean_or_nan(values: List[float]) -> float:
    values = [v for v in values if pd.notna(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _collect_violin_data(
    datasets: List[str],
    mu_values: List[float],
    results_dir: Path,
) -> pd.DataFrame:
    metric_map = {
        "loss": ("with_constraint_loss_mean", "without_constraint_loss_mean"),
        "pos_loss": ("with_constraint_loss_pos_mean", "without_constraint_loss_pos_mean"),
        "rset_size": ("with_constraint_rset_size", "without_constraint_rset_size"),
        "pos_sparsity": ("with_constraint_sparsity_pos_mean", "without_constraint_sparsity_pos_mean"),
    }
    records = []
    for dataset in datasets:
        detailed_df = _load_detailed_results(dataset, results_dir)
        if detailed_df.empty:
            continue
        for mu in mu_values:
            bc_df = _filter_bc(detailed_df, mu)
            if bc_df.empty:
                continue
            for metric_key, (with_col, without_col) in metric_map.items():
                for label, col_name in (
                    ("GraviTree", with_col),
                    ("Regular Trees", without_col),
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


def _plot_violin_summary(df: pd.DataFrame, mu_values: List[float], output_path: Path) -> None:
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
        fig.suptitle(f"GraviTree (μ = {mu_key}) vs Regular Trees", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        mu_output = output_path.with_name(f"{output_path.stem}_mu_{mu_key}{output_path.suffix}")
        fig.savefig(mu_output, dpi=300)
        print(f"Saved violin plot to {mu_output}")


def summarize_dataset(
    dataset: str,
    mu_values: List[float],
    ft_dir: Path,
    regular_dir: Path,
    num_trials: int | None,
) -> Dict[str, float]:
    del ft_dir, num_trials
    row = {"dataset": dataset}
    detailed_df = _load_detailed_results(dataset, regular_dir)
    for mu in mu_values:
        bc_df = _filter_bc(detailed_df, mu)
        ft_df = bc_df
        reg_df = bc_df
        ft_rset_mean, ft_rset_std = _mean_std(
            ft_df.get("with_constraint_rset_size", pd.Series(dtype=float)).tolist()
        )
        ft_sparsity_mean, ft_sparsity_std = _mean_std(
            ft_df.get("with_constraint_sparsity_pos_mean", pd.Series(dtype=float)).tolist()
        )
        ft_loss_pos_mean, ft_loss_pos_std = _mean_std(
            ft_df.get("with_constraint_loss_pos_mean", pd.Series(dtype=float)).tolist()
        )
        ft_loss_mean, ft_loss_std = _mean_std(
            ft_df.get("with_constraint_loss_mean", pd.Series(dtype=float)).tolist()
        )
        ft_time_mean, ft_time_std = _mean_std(
            ft_df.get("with_constraint_time", pd.Series(dtype=float)).tolist()
        )

        reg_rset_mean, reg_rset_std = _mean_std(
            reg_df.get("without_constraint_rset_size", pd.Series(dtype=float)).tolist()
        )
        reg_sparsity_mean, reg_sparsity_std = _mean_std(
            reg_df.get("without_constraint_sparsity_pos_mean", pd.Series(dtype=float)).tolist()
        )
        reg_loss_pos_mean, reg_loss_pos_std = _mean_std(
            reg_df.get("without_constraint_loss_pos_mean", pd.Series(dtype=float)).tolist()
        )
        reg_loss_mean, reg_loss_std = _mean_std(
            reg_df.get("without_constraint_loss_mean", pd.Series(dtype=float)).tolist()
        )
        reg_time_mean, reg_time_std = _mean_std(
            reg_df.get("without_constraint_time", pd.Series(dtype=float)).tolist()
        )

        row[f"mu_{mu}_ft_rset_size_mean"] = ft_rset_mean
        row[f"mu_{mu}_ft_rset_size_std"] = ft_rset_std
        row[f"mu_{mu}_ft_sparsity_pos_mean"] = ft_sparsity_mean
        row[f"mu_{mu}_ft_sparsity_pos_std"] = ft_sparsity_std
        row[f"mu_{mu}_ft_loss_pos_mean"] = ft_loss_pos_mean
        row[f"mu_{mu}_ft_loss_pos_std"] = ft_loss_pos_std
        row[f"mu_{mu}_ft_loss_mean"] = ft_loss_mean
        row[f"mu_{mu}_ft_loss_std"] = ft_loss_std
        row[f"mu_{mu}_ft_time_mean"] = ft_time_mean
        row[f"mu_{mu}_ft_time_std"] = ft_time_std

        row[f"mu_{mu}_reg_rset_size_mean"] = reg_rset_mean
        row[f"mu_{mu}_reg_rset_size_std"] = reg_rset_std
        row[f"mu_{mu}_reg_sparsity_pos_mean"] = reg_sparsity_mean
        row[f"mu_{mu}_reg_sparsity_pos_std"] = reg_sparsity_std
        row[f"mu_{mu}_reg_loss_pos_mean"] = reg_loss_pos_mean
        row[f"mu_{mu}_reg_loss_pos_std"] = reg_loss_pos_std
        row[f"mu_{mu}_reg_loss_mean"] = reg_loss_mean
        row[f"mu_{mu}_reg_loss_std"] = reg_loss_std
        row[f"mu_{mu}_reg_time_mean"] = reg_time_mean
        row[f"mu_{mu}_reg_time_std"] = reg_time_std

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Falling Trees vs Regular Trees across mu values."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of datasets (e.g., compas,bar,bar7)",
    )
    parser.add_argument(
        "--mu_values",
        type=str,
        default="0.01,0.1",
        help="Comma-separated mu values (default: 0.01,0.1)",
    )
    parser.add_argument(
        "--ft_dir",
        type=str,
        default=None,
        help="Directory for falling_trees_vs_frame_runtime results (overrides --max_len)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1,
        help="Max rule length to select results dir when --ft_dir is not set (default: 1)",
    )
    parser.add_argument(
        "--regular_dir",
        type=str,
        default="falling_trees_vs_regular_trees_results",
        help="Directory for falling_trees_vs_regular_trees results",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=None,
        help="Number of split indices to consider (default: infer from files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mu_summary.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default="mu_summary_violin.png",
        help="Output filename for the violin plot summary",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    mu_values = [float(x.strip()) for x in args.mu_values.split(",") if x.strip()]

    if args.ft_dir:
        ft_dir = Path(args.ft_dir)
    else:
        ft_dir = Path(args.regular_dir)
    regular_dir = Path(args.regular_dir)

    rows = []
    for dataset in datasets:
        rows.append(
            summarize_dataset(
                dataset,
                mu_values,
                ft_dir,
                regular_dir,
                args.num_trials,
            )
        )

    out_path = Path(args.output)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")

    violin_df = _collect_violin_data(datasets, mu_values, regular_dir)
    _plot_violin_summary(violin_df, mu_values, Path(args.plot_output))


if __name__ == "__main__":
    main()

