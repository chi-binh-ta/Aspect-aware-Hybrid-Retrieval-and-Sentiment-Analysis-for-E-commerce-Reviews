"""Generate Module 5 report-ready figures with matplotlib only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def save_module2_comparison(project_dir: Path, figure_dir: Path) -> Path | None:
    path = project_dir / "results/module2/metrics_summary.csv"
    if not path.exists():
        print("[module5_figures] Skip Module 2 comparison: missing metrics_summary.csv")
        return None
    df = pd.read_csv(path)
    required = {"model_name", "split", "accuracy", "macro_f1"}
    if df.empty or not required.issubset(df.columns):
        return None
    test = df[(df["split"] == "test") & (df["model_name"] != "majority_class")].copy()
    if test.empty:
        return None
    x = range(len(test))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], test["accuracy"], width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], test["macro_f1"], width=width, label="Macro-F1")
    plt.xticks(list(x), test["model_name"], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Module 2 Model Comparison on Test Set")
    plt.legend()
    plt.tight_layout()
    output = figure_dir / "module2_model_comparison.png"
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def save_module3_sentiment(project_dir: Path, figure_dir: Path) -> Path | None:
    path = project_dir / "results/module3/overall_sentiment_distribution.csv"
    if not path.exists():
        print("[module5_figures] Skip Module 3 sentiment chart: missing overall_sentiment_distribution.csv")
        return None
    df = pd.read_csv(path)
    sentiment_col = None
    for candidate in ["predicted_sentiment", "sentiment"]:
        if candidate in df.columns:
            sentiment_col = candidate
            break
    if df.empty or sentiment_col is None or "count" not in df.columns:
        return None
    plt.figure(figsize=(7, 5))
    plt.bar(df[sentiment_col], df["count"])
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Count")
    plt.title("Module 3 RAG Corpus Sentiment Distribution")
    plt.tight_layout()
    output = figure_dir / "module3_sentiment_distribution.png"
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def save_manual_precision(project_dir: Path, figure_dir: Path) -> Path | None:
    path = project_dir / "results/module5/manual_precision_overall.csv"
    if not path.exists():
        print("[module5_figures] Skip manual Precision@k chart: no computed manual precision.")
        return None
    df = pd.read_csv(path)
    if df.empty or not {"k", "average_precision_at_k"}.issubset(df.columns):
        return None
    plt.figure(figsize=(7, 5))
    plt.bar(df["k"].astype(str), df["average_precision_at_k"])
    plt.ylim(0, 1)
    plt.xlabel("k")
    plt.ylabel("Average Precision@k")
    plt.title("Manual Retrieval Precision@k")
    plt.tight_layout()
    output = figure_dir / "manual_precision_at_k.png"
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def main() -> None:
    project_dir = resolve_project_dir()
    figure_dir = project_dir / "figures" / "module5"
    figure_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        save_module2_comparison(project_dir, figure_dir),
        save_module3_sentiment(project_dir, figure_dir),
        save_manual_precision(project_dir, figure_dir),
    ]
    for output in outputs:
        if output is not None:
            print(f"[module5_figures] Saved: {output}")


if __name__ == "__main__":
    main()
