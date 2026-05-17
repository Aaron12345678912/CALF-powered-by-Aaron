#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Optional

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<model>[^_]+)_"
    r"(?P<seq_len>\d+)_"
    r"(?P<pred_len>\d+)_"
    r"(?P<d_model>\d+)_"
    r"(?P<n_heads>\d+)_"
    r"(?P<learning_rate>[0-9eE+\-.]+)_"
    r"(?P<feature_w>[0-9eE+\-.]+)_"
    r"ow(?P<output_w>[0-9eE+\-.]+)_"
    r"r(?P<r>\d+)_"
    r"la(?P<lora_alpha>\d+)_"
    r"ld(?P<lora_dropout>[0-9eE+\-.]+)_"
    r"dff(?P<d_ff>\d+)_"
    r"(?P<random_seed>\d+)\.logs$"
)

MSE_PATTERN = re.compile(r"mse[:\s]*([0-9eE+\-.]+)", re.IGNORECASE)
MAE_PATTERN = re.compile(r"mae[:\s]*([0-9eE+\-.]+)", re.IGNORECASE)


def _to_number(x: str):
    if any(ch in x for ch in [".", "e", "E"]):
        return float(x)
    return int(x)


def extract_metrics(log_path: str) -> Dict[str, Optional[float]]:
    mse_val = None
    mae_val = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mse_match = MSE_PATTERN.search(line)
            mae_match = MAE_PATTERN.search(line)
            if mse_match:
                mse_val = float(mse_match.group(1))
            if mae_match:
                mae_val = float(mae_match.group(1))

    return {"mse": mse_val, "mae": mae_val}


def parse_log_file(log_path: str) -> Optional[Dict]:
    base = os.path.basename(log_path)
    m = FILENAME_PATTERN.match(base)
    if not m:
        return None

    row = {k: _to_number(v) for k, v in m.groupdict().items()}
    row["log_file"] = base
    row["log_path"] = os.path.abspath(log_path)

    row.update(extract_metrics(log_path))
    row["has_metric"] = row["mse"] is not None and row["mae"] is not None

    return row


def build_summary(logs_dir: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for name in os.listdir(logs_dir):
        if not name.endswith(".logs"):
            continue
        full = os.path.join(logs_dir, name)
        if not os.path.isfile(full):
            continue
        parsed = parse_log_file(full)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "mse" in df.columns:
        df = df.sort_values(by=["mse", "mae"], na_position="last").reset_index(drop=True)
        df["rank_by_mse"] = df["mse"].rank(method="min", na_option="bottom")

    return df


def aggregate_by_param_combo(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "seq_len",
        "pred_len",
        "learning_rate",
        "d_model",
        "n_heads",
        "d_ff",
        "feature_w",
        "output_w",
        "r",
        "lora_alpha",
        "lora_dropout",
    ]

    metric_df = df[df["has_metric"]].copy()
    if metric_df.empty:
        return pd.DataFrame(columns=key_cols + ["runs", "mse_mean", "mse_std", "mae_mean", "mae_std", "best_seed", "best_mse", "best_mae"])

    grouped = metric_df.groupby(key_cols, dropna=False)
    agg = grouped.agg(
        runs=("random_seed", "count"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
    ).reset_index()

    best_rows = metric_df.sort_values(["mse", "mae"], ascending=[True, True]).groupby(key_cols, as_index=False).first()
    best_rows = best_rows[key_cols + ["random_seed", "mse", "mae"]].rename(
        columns={"random_seed": "best_seed", "mse": "best_mse", "mae": "best_mae"}
    )

    merged = agg.merge(best_rows, on=key_cols, how="left")
    merged = merged.sort_values(["pred_len", "mse_mean", "mae_mean"], ascending=[True, True, True]).reset_index(drop=True)
    return merged


def topk_by_pred_len(combo_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if combo_df.empty:
        return combo_df

    output = []
    for pred_len, part in combo_df.groupby("pred_len"):
        top = part.nsmallest(top_k, columns=["mse_mean", "mae_mean"]).copy()
        top.insert(0, "group_pred_len", pred_len)
        output.append(top)

    return pd.concat(output, axis=0, ignore_index=True) if output else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="提取 Solar 日志中的 mse/mae 和参数组合并导出 Excel")
    parser.add_argument("--logs_dir", type=str, default="logs/CALF/Solar", help="日志目录")
    parser.add_argument(
        "--excel_path",
        type=str,
        default="logs/CALF/Solar/solar_metrics_summary.xlsx",
        help="输出 Excel 路径",
    )
    parser.add_argument("--top_k", type=int, default=5, help="每个 pred_len 导出 top-k 参数组合")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        raise FileNotFoundError(f"日志目录不存在: {args.logs_dir}")

    df = build_summary(args.logs_dir)
    if df.empty:
        raise RuntimeError(f"未在 {args.logs_dir} 中解析到符合命名规则的 .logs 文件")

    combo_df = aggregate_by_param_combo(df)
    topk_df = topk_by_pred_len(combo_df, args.top_k)

    os.makedirs(os.path.dirname(args.excel_path), exist_ok=True)

    with pd.ExcelWriter(args.excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_runs")
        combo_df.to_excel(writer, index=False, sheet_name="combo_stats")
        topk_df.to_excel(writer, index=False, sheet_name="topk_by_pred_len")

    valid_cnt = int(df["has_metric"].sum())
    print(f"已导出: {args.excel_path}")
    print(f"日志总数: {len(df)}, 含 mse/mae 的日志数: {valid_cnt}")


if __name__ == "__main__":
    main()
