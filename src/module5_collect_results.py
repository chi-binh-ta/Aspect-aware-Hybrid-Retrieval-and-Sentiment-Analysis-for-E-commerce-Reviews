"""Collect final project metrics from Modules 2-4.

Module 5 is read-only with respect to earlier modules: it aggregates existing
CSV/JSON outputs and writes report-ready summaries under results/module5.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

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


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_metric(rows: list[dict[str, Any]], module: str, metric: str, value: Any, source: str, **extra: Any) -> None:
    rows.append(
        {
            "module": module,
            "metric": metric,
            "value": value,
            "source_file": source,
            **extra,
        }
    )


def best_module2_model(metrics: pd.DataFrame | None) -> tuple[str | None, float | None, float | None]:
    if metrics is None or metrics.empty:
        return None, None, None
    if not {"model_name", "split", "macro_f1"}.issubset(metrics.columns):
        return None, None, None

    valid = metrics[(metrics["split"] == "valid") & (metrics["model_name"] != "majority_class")].copy()
    if valid.empty:
        return None, None, None
    best_row = valid.sort_values("macro_f1", ascending=False).iloc[0]
    best_model = str(best_row["model_name"])
    valid_macro_f1 = float(best_row["macro_f1"])

    test_macro_f1 = None
    test_rows = metrics[(metrics["split"] == "test") & (metrics["model_name"] == best_model)]
    if not test_rows.empty and pd.notna(test_rows.iloc[0].get("macro_f1")):
        test_macro_f1 = float(test_rows.iloc[0]["macro_f1"])
    return best_model, test_macro_f1, valid_macro_f1


def main() -> None:
    project_dir = resolve_project_dir()
    results_dir = project_dir / "results" / "module5"
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "module2_metrics": project_dir / "results" / "module2" / "metrics_summary.csv",
        "module2_svm_report": project_dir / "results" / "module2" / "linear_svm_classification_report.csv",
        "module2_logreg_report": project_dir / "results" / "module2" / "logistic_regression_classification_report.csv",
        "module3_summary": project_dir / "results" / "module3" / "rag_sentiment_summary.json",
        "module3_overall": project_dir / "results" / "module3" / "overall_sentiment_distribution.csv",
        "module3_rating": project_dir / "results" / "module3" / "rating_sentiment_crosstab.csv",
        "module4_rag": project_dir / "results" / "module4" / "rag_outputs.json",
        "module4_eval": project_dir / "results" / "module4" / "retrieval_eval_summary.csv",
        "module4_manual": project_dir / "results" / "module4" / "manual_precision_at_k_sheet.csv",
    }
    missing_optional_files = [str(path.relative_to(project_dir)) for path in paths.values() if not path.exists()]

    module2_metrics = read_csv(paths["module2_metrics"])
    module2_svm_report = read_csv(paths["module2_svm_report"])
    module2_logreg_report = read_csv(paths["module2_logreg_report"])
    module3_summary = read_json(paths["module3_summary"])
    module3_overall = read_csv(paths["module3_overall"])
    module4_rag = read_json(paths["module4_rag"])
    module4_eval = read_csv(paths["module4_eval"])
    module4_manual = read_csv(paths["module4_manual"])

    rows: list[dict[str, Any]] = []

    best_model, best_test_macro_f1, best_valid_macro_f1 = best_module2_model(module2_metrics)
    if module2_metrics is not None and not module2_metrics.empty:
        for _, row in module2_metrics.iterrows():
            for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
                if metric in row and pd.notna(row[metric]):
                    add_metric(
                        rows,
                        "module2",
                        metric,
                        float(row[metric]),
                        str(paths["module2_metrics"].relative_to(project_dir)),
                        model_name=row.get("model_name"),
                        split=row.get("split"),
                    )

    for report_name, report_df, report_path in [
        ("linear_svm", module2_svm_report, paths["module2_svm_report"]),
        ("logistic_regression", module2_logreg_report, paths["module2_logreg_report"]),
    ]:
        if report_df is not None and not report_df.empty and {"split", "label", "f1_score"}.issubset(report_df.columns):
            test_rows = report_df[report_df["split"] == "test"]
            for _, row in test_rows.iterrows():
                add_metric(
                    rows,
                    "module2",
                    "classification_report_f1",
                    float(row["f1_score"]) if pd.notna(row["f1_score"]) else None,
                    str(report_path.relative_to(project_dir)),
                    model_name=report_name,
                    split=row.get("split"),
                    label=row.get("label"),
                )

    module3_total_reviews = None
    module3_sentiment_distribution = None
    if module3_summary:
        module3_total_reviews = (
            module3_summary.get("total_documents")
            or module3_summary.get("rag_total_documents")
            or module3_summary.get("total_reviews")
        )
        module3_sentiment_distribution = (
            module3_summary.get("overall_sentiment_distribution")
            or module3_summary.get("sentiment_distribution")
        )
        if module3_total_reviews is not None:
            add_metric(rows, "module3", "total_reviews", int(module3_total_reviews), str(paths["module3_summary"].relative_to(project_dir)))
    elif module3_overall is not None and not module3_overall.empty:
        if {"predicted_sentiment", "count"}.issubset(module3_overall.columns):
            module3_sentiment_distribution = {
                str(row["predicted_sentiment"]): int(row["count"])
                for _, row in module3_overall.iterrows()
            }
            module3_total_reviews = int(module3_overall["count"].sum())

    if module3_overall is not None and not module3_overall.empty and {"predicted_sentiment", "count"}.issubset(module3_overall.columns):
        for _, row in module3_overall.iterrows():
            add_metric(
                rows,
                "module3",
                f"sentiment_count_{row['predicted_sentiment']}",
                int(row["count"]),
                str(paths["module3_overall"].relative_to(project_dir)),
            )

    module4_num_eval_queries = None
    if module4_eval is not None and not module4_eval.empty:
        if "query_id" in module4_eval.columns:
            module4_num_eval_queries = int(module4_eval["query_id"].nunique())
            add_metric(rows, "module4", "num_eval_queries", module4_num_eval_queries, str(paths["module4_eval"].relative_to(project_dir)))
        for k, group in module4_eval.groupby("k") if "k" in module4_eval.columns else []:
            if "num_results" in group.columns:
                add_metric(
                    rows,
                    "module4",
                    "avg_num_results",
                    float(group["num_results"].mean()),
                    str(paths["module4_eval"].relative_to(project_dir)),
                    k=int(k),
                )
            if "avg_score" in group.columns:
                add_metric(
                    rows,
                    "module4",
                    "avg_retrieval_score",
                    float(group["avg_score"].mean()),
                    str(paths["module4_eval"].relative_to(project_dir)),
                    k=int(k),
                )

    module4_rag_query = module4_rag.get("query") if module4_rag else None
    module4_strong_negative_count = module4_rag.get("strong_negative_count") if module4_rag else None
    module4_noise_filtered_count = module4_rag.get("noise_filtered_count") if module4_rag else None
    if module4_rag:
        for metric in ["retrieved_rows", "strong_negative_count", "weak_or_mixed_count", "noise_filtered_count"]:
            if metric in module4_rag:
                add_metric(rows, "module4", metric, module4_rag[metric], str(paths["module4_rag"].relative_to(project_dir)))

    if module4_manual is not None and not module4_manual.empty:
        labeled = module4_manual.get("is_relevant_manual", pd.Series(dtype=object)).notna().sum()
        add_metric(rows, "module4", "manual_label_rows_present", int(labeled), str(paths["module4_manual"].relative_to(project_dir)))

    metrics_df = pd.DataFrame(rows)
    metrics_path = results_dir / "final_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary = {
        "module2_best_model": best_model,
        "module2_best_macro_f1": best_test_macro_f1 if best_test_macro_f1 is not None else best_valid_macro_f1,
        "module2_best_valid_macro_f1": best_valid_macro_f1,
        "module3_total_reviews": module3_total_reviews,
        "module3_sentiment_distribution": module3_sentiment_distribution,
        "module4_num_eval_queries": module4_num_eval_queries,
        "module4_rag_query": module4_rag_query,
        "module4_strong_negative_count": module4_strong_negative_count,
        "module4_noise_filtered_count": module4_noise_filtered_count,
        "missing_optional_files": missing_optional_files,
    }
    summary_path = results_dir / "final_project_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[module5_collect] Saved: {metrics_path}")
    print(f"[module5_collect] Saved: {summary_path}")
    if missing_optional_files:
        print("[module5_collect] Missing optional files:")
        for path in missing_optional_files:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
