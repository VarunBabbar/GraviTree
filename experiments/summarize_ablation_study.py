"""
Summarize ablation study detailed results into per-branching-cost CSVs.
Scans ablation_study_results_mu_* directories and aggregates per dataset/config.
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


def _summarize_group(df: pd.DataFrame) -> Dict[str, float]:
    row: Dict[str, float] = {
        "dataset": df["dataset"].iloc[0],
        "config": df["config"].iloc[0] if "config" in df.columns else "all",
    }
    if "branching_cost" in df.columns:
        row["branching_cost"] = df["branching_cost"].iloc[0]

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"dataset", "config", "branching_cost"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in numeric_cols:
        values = df[col].dropna().tolist()
        row[f"{col}_mean"] = _safe_mean(values)
        row[f"{col}_std"] = _safe_std(values)
    return row


def summarize_file(df: pd.DataFrame) -> List[Dict[str, float]]:
    if df.empty or "dataset" not in df.columns:
        return []
    if "config" in df.columns:
        rows = []
        for _, group in df.groupby(["dataset", "config"], dropna=False):
            rows.append(_summarize_group(group))
        return rows
    return [_summarize_group(df)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ablation study detailed results by branching cost."
    )
    parser.add_argument(
        "--results_base_dir",
        type=str,
        default=".",
        help="Base directory containing ablation_study_results_mu_* directories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory for summary CSVs",
    )

    args = parser.parse_args()
    results_base_dir = Path(args.results_base_dir)
    if not results_base_dir.exists():
        raise FileNotFoundError(f"Results base directory not found: {results_base_dir}")

    result_dirs = sorted(results_base_dir.glob("ablation_study_results_mu_*"))
    if not result_dirs:
        raise FileNotFoundError(
            "No result directories found matching pattern: ablation_study_results_mu_*"
        )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for result_dir in result_dirs:
        mu_str = result_dir.name.split("ablation_study_results_mu_")[-1]
        csv_files = sorted(result_dir.glob("*_ablation_detailed_results.csv"))
        if not csv_files:
            continue

        rows: List[Dict[str, float]] = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            rows.extend(summarize_file(df))

        if not rows:
            continue

        summary_df = pd.DataFrame(rows)
        summary_df = summary_df[["dataset","config","branching_cost","rset_size_mean", "rset_size_std", "avg_loss_mean", "avg_loss_std", "avg_sparsity_mean", "avg_sparsity_std", "runtime_mean", "runtime_std"]]
        out_base = output_dir if output_dir else result_dir
        out_path = out_base / f"ablation_study_summary_mu_{mu_str}.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"Saved summary to {out_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()

