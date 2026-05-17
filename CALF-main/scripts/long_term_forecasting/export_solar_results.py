#!/usr/bin/env python3
import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<model>[^_]+)_"
    r"(?P<seq_len>\d+)_"
    r"(?P<pred_len>\d+)_"
    r"(?P<d_model>\d+)_"
    r"(?P<n_heads>\d+)_"
    r"(?P<learning_rate>[\d\.eE+-]+)_"
    r"(?P<feature_w>[\d\.eE+-]+)_"
    r"ow(?P<output_w>[\d\.eE+-]+)_"
    r"r(?P<r>\d+)_"
    r"la(?P<lora_alpha>\d+)_"
    r"ld(?P<lora_dropout>[\d\.eE+-]+)_"
    r"dff(?P<d_ff>\d+)_"
    r"(?P<random_seed>\d+)\.logs$"
)

METRIC_PATTERN = re.compile(
    r"mse\s*:\s*(?P<mse>[\d\.eE+-]+)\s*,\s*mae\s*:\s*(?P<mae>[\d\.eE+-]+)",
    re.IGNORECASE,
)


def _to_number(v: str):
    if re.fullmatch(r"\d+", v):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def parse_log_file(log_path: str) -> Optional[Dict]:
    base = os.path.basename(log_path)
    m = FILENAME_PATTERN.match(base)
    if not m:
        return None

    info = {k: _to_number(v) for k, v in m.groupdict().items()}

    last_metrics: Optional[Tuple[float, float]] = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                mm = METRIC_PATTERN.search(line)
                if mm:
                    last_metrics = (float(mm.group("mse")), float(mm.group("mae")))
    except OSError:
        return None

    if last_metrics is None:
        return None

    info["mse"] = last_metrics[0]
    info["mae"] = last_metrics[1]
    info["log_file"] = log_path
    return info


def build_tables(logs_dir: str):
    files = sorted(glob.glob(os.path.join(logs_dir, "*.logs")))
    rows: List[Dict] = []
    for p in files:
        row = parse_log_file(p)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(rows)
    detail_df = detail_df.sort_values(["pred_len", "mse", "mae", "random_seed"], ascending=[True, True, True, True])

    group_cols = [
        "model",
        "seq_len",
        "pred_len",
        "d_model",
        "n_heads",
        "learning_rate",
        "feature_w",
        "output_w",
        "r",
        "lora_alpha",
        "lora_dropout",
        "d_ff",
    ]

    summary_df = (
        detail_df.groupby(group_cols, as_index=False)
        .agg(
            seed_count=("random_seed", "count"),
            avg_mse=("mse", "mean"),
            avg_mae=("mae", "mean"),
            min_mse=("mse", "min"),
            min_mae=("mae", "min"),
            std_mse=("mse", "std"),
            std_mae=("mae", "std"),
        )
        .sort_values(["avg_mse", "avg_mae"], ascending=[True, True])
    )

    best_idx = detail_df.groupby(group_cols)["mse"].idxmin()
    best_seed_df = detail_df.loc[best_idx, group_cols + ["random_seed", "mse", "mae", "log_file"]].rename(
        columns={
            "random_seed": "best_seed",
            "mse": "best_seed_mse",
            "mae": "best_seed_mae",
            "log_file": "best_seed_log",
        }
    )

    summary_df = summary_df.merge(best_seed_df, on=group_cols, how="left")

    return detail_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="Export Solar logs to Excel with mse/mae and parameter combinations")
    parser.add_argument("--logs_dir", type=str, default="logs/CALF/Solar", help="Directory containing .logs files")
    parser.add_argument(
        "--out_excel",
        type=str,
        default="logs/CALF/Solar/solar_metrics_summary.xlsx",
        help="Output Excel file path",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Top-k rows (sorted by avg_mse) for a dedicated sheet")
    args = parser.parse_args()

    detail_df, summary_df = build_tables(args.logs_dir)
    if detail_df.empty:
        print(f"No valid logs found in: {args.logs_dir}")
        return

    os.makedirs(os.path.dirname(args.out_excel), exist_ok=True)

    top_k = max(1, args.top_k)
    top_df = summary_df.head(top_k)

    with pd.ExcelWriter(args.out_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_by_combo", index=False)
        top_df.to_excel(writer, sheet_name=f"top{top_k}", index=False)
        detail_df.to_excel(writer, sheet_name="all_runs", index=False)

    print(f"Exported {len(detail_df)} runs and {len(summary_df)} parameter combos -> {args.out_excel}")


if __name__ == "__main__":
    main()
