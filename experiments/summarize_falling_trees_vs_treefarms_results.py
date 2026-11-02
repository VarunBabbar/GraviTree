"""
Summarize Falling Trees vs TREEFARMS detailed results into a single CSV.
"""

from __future__ import annotations

import argparse
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
    preferred_cols = [
        "falling_trees_time",
        "treefarms_time",
        "falling_trees_rset_size",
        "treefarms_rset_size",
        "falling_trees_sparsity_mean",
        "treefarms_sparsity_mean",
        "falling_trees_sparsity_pos_mean",
        "treefarms_sparsity_pos_mean",
        "falling_trees_loss_mean",
        "treefarms_loss_mean",
        "falling_trees_loss_pos_mean",
        "treefarms_loss_pos_mean",
    ]
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"dataset", "split_idx"}
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
        description="Summarize Falling Trees vs TREEFARMS detailed results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="falling_trees_vs_treefarms_results",
        help="Directory containing *_full_detailed_results.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="falling_trees_vs_treefarms_summary.csv",
        help="Output CSV filename",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows: List[Dict[str, float]] = []
    for path in sorted(results_dir.glob("*_full_detailed_results.csv")):
        df = pd.read_csv(path)
        if df.empty or "dataset" not in df.columns:
            continue
        rows.append(summarize_dataset(df))

    out_path = Path(args.output)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()

