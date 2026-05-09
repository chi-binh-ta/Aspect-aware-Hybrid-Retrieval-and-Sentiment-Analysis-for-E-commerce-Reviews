"""Compute manual retrieval precision@k when labels are available."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def resolve_project_dir() -> Path:
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def normalize_manual_label(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text in {"0", "0.0", "false", "False", "FALSE"}:
        return 0.0
    if text in {"1", "1.0", "true", "True", "TRUE"}:
        return 1.0
    return None


def main() -> None:
    project_dir = resolve_project_dir()
    results_dir = project_dir / "results" / "module5"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_path = project_dir / "results" / "module4" / "manual_precision_at_k_sheet.csv"
    summary_path = results_dir / "manual_precision_summary.csv"
    overall_path = results_dir / "manual_precision_overall.csv"

    if not input_path.exists():
        summary = pd.DataFrame(
            [
                {
                    "status": "manual sheet missing",
                    "message": "results/module4/manual_precision_at_k_sheet.csv was not found",
                }
            ]
        )
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[module5_manual] Warning: missing {input_path}")
        print(f"[module5_manual] Saved: {summary_path}")
        return

    df = pd.read_csv(input_path)
    if "is_relevant_manual" not in df.columns:
        df["is_relevant_manual"] = pd.NA
    df["manual_label"] = df["is_relevant_manual"].apply(normalize_manual_label)
    labeled = df[df["manual_label"].notna()].copy()

    if labeled.empty:
        summary = pd.DataFrame(
            [
                {
                    "status": "manual labels not provided",
                    "message": "Fill is_relevant_manual with 0/1 to compute Precision@k.",
                    "input_path": str(input_path),
                    "num_rows": int(len(df)),
                }
            ]
        )
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        if overall_path.exists():
            overall_path.unlink()
        print("[module5_manual] Warning: manual labels not provided.")
        print("[module5_manual] Fill is_relevant_manual with 0/1, then rerun this script.")
        print(f"[module5_manual] Saved: {summary_path}")
        return

    required = {"query_id", "query", "k", "manual_label"}
    missing = required - set(labeled.columns)
    if missing:
        raise KeyError(f"manual_precision_at_k_sheet.csv is missing columns: {sorted(missing)}")

    grouped = (
        labeled.groupby(["query_id", "query", "k"], dropna=False)
        .agg(
            precision_at_k=("manual_label", "mean"),
            num_labeled=("manual_label", "size"),
            num_relevant=("manual_label", "sum"),
        )
        .reset_index()
    )
    grouped["status"] = "computed"
    grouped.to_csv(summary_path, index=False, encoding="utf-8-sig")

    overall = (
        grouped.groupby("k", dropna=False)
        .agg(
            average_precision_at_k=("precision_at_k", "mean"),
            num_queries=("query_id", "nunique"),
            num_labeled_rows=("num_labeled", "sum"),
        )
        .reset_index()
    )
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    print(f"[module5_manual] Saved: {summary_path}")
    print(f"[module5_manual] Saved: {overall_path}")


if __name__ == "__main__":
    main()

