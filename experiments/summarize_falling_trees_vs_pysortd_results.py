"""
Summarize Falling Trees vs pysortd detailed results into a single CSV.
Aggregates results from all branching cost directories.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def summarize_dataset(df: pd.DataFrame) -> Dict[str, float]:
    row = {"dataset": df["dataset"].iloc[0]}
    # Include branching_cost if present
    if "branching_cost" in df.columns:
        row["branching_cost"] = df["branching_cost"].iloc[0]
    preferred_cols = [
        "falling_trees_time",
        "pysortd_time",
        "falling_trees_rset_size",
        "pysortd_rset_size",
        "falling_trees_sparsity_mean",
        "pysortd_sparsity_mean",
        "falling_trees_sparsity_pos_mean",
        "pysortd_sparsity_pos_mean",
        "falling_trees_loss_mean",
        "pysortd_loss_mean",
        "falling_trees_loss_pos_mean",
        "pysortd_loss_pos_mean",
    ]
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"dataset", "split_idx", "branching_cost"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in preferred_cols:
        if col not in numeric_cols and col in df.columns:
            numeric_cols.append(col)
    for col in numeric_cols:
        values = df[col].dropna().tolist()
        row[f"{col}_mean"] = _safe_mean(values)
        row[f"{col}_std"] = _safe_std(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Falling Trees vs pysortd detailed results."
    )
    parser.add_argument(
        "--results_base_dir",
        type=str,
        default=".",
        help="Base directory containing falling_trees_vs_pysortd_results_* directories",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.02,
        help="Lambda (regularization) parameter to filter results",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Epsilon (Rashomon budget) parameter to filter results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV filename (default: falling_trees_vs_pysortd_summary_{lam}_{eps}.csv)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth of trees to filter results",
    )
    parser.add_argument(
        "--branching-cost",
        type=float,
        default=0.01,
        help="Branching cost to filter results",
    )

    args = parser.parse_args()
    results_base_dir = Path(args.results_base_dir)
    if not results_base_dir.exists():
        raise FileNotFoundError(f"Results base directory not found: {results_base_dir}")

    # Find all directories matching the pattern falling_trees_vs_pysortd_results_{lam}_{eps}_{mu}
    pattern = f"falling_trees_vs_pysortd_results_{args.lam}_{args.eps}_{args.branching_cost}"
    result_dirs = sorted(results_base_dir.glob(pattern))
    
    if not result_dirs:
        raise FileNotFoundError(
            f"No result directories found matching pattern: {pattern}"
        )

    print(f"Found {len(result_dirs)} result directories")
    for dir_path in result_dirs:
        print(f"  - {dir_path}")

    rows: List[Dict[str, float]] = []
    for result_dir in result_dirs:
        # Find CSV files in this directory
        csv_files = sorted(result_dir.glob("*_full_detailed_results.csv"))
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            if df.empty or "dataset" not in df.columns:
                continue
            
            # Extract branching cost from directory name
            dir_name = result_dir.name
            # Format: falling_trees_vs_pysortd_results_{lam}_{eps}_{mu}
            # Example: falling_trees_vs_pysortd_results_0.02_0.01_0.005
            # Split by "_" and take the last part as mu
            parts = dir_name.split("_")
            if len(parts) >= 6:
                try:
                    # Last part should be the branching cost (mu)
                    mu_str = parts[-1]
                    mu = float(mu_str)
                    df["branching_cost"] = mu
                except (ValueError, IndexError):
                    # If parsing fails, try to extract from the full directory name
                    # Pattern: falling_trees_vs_pysortd_results_{lam}_{eps}_{mu}
                    match = re.search(r'falling_trees_vs_pysortd_results_[\d.]+_[\d.]+_([\d.]+)$', dir_name)
                    if match:
                        try:
                            mu = float(match.group(1))
                            df["branching_cost"] = mu
                        except ValueError:
                            pass
            
            rows.append(summarize_dataset(df))

    if not rows:
        raise ValueError("No valid results found to summarize")

    out_path = Path(args.output) if args.output else Path(f"falling_trees_vs_pysortd_summary_{args.lam}_{args.eps}_{args.branching_cost}_{args.depth}.csv")
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")
    print(f"Summary contains {len(summary_df)} rows")


if __name__ == "__main__":
    main()

