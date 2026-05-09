"""Prepare the RAG review corpus for Module 4 retrieval.

This script creates data/processed/rag_corpus_module4.csv. If Module 3 has
already enriched the RAG corpus with predicted sentiment, those fields are
preserved. Otherwise it loads the Module 2 sentiment model and predicts
sentiment for each review.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


LABELS = ["negative", "neutral", "positive"]
TEXT_PRIORITY = ["comment", "text", "clean_text", "review", "retrieval_text"]
MODEL_CANDIDATES = [
    "models/module2/tfidf_linearsvm.joblib",
    "models/module2/best_model.joblib",
]


def resolve_project_dir(project_dir: str | None = None) -> Path:
    if project_dir:
        return Path(project_dir).expanduser().resolve()
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_path(project_dir: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_dir / path


def ensure_dirs(project_dir: Path) -> dict[str, Path]:
    dirs = {
        "processed": project_dir / "data" / "processed",
        "results": project_dir / "results" / "module4",
        "models": project_dir / "models" / "module4",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def is_empty_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def first_existing_input(project_dir: Path) -> Path:
    candidates = [
        project_dir / "data" / "processed" / "rag_corpus_with_sentiment.csv",
        project_dir / "data" / "processed" / "rag_corpus.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No RAG corpus found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def choose_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_PRIORITY:
        if column in df.columns and not is_empty_text(df[column]).all():
            return column
    raise KeyError(f"No usable text column found. Tried: {TEXT_PRIORITY}")


def find_model_path(project_dir: Path) -> Path:
    for relative_path in MODEL_CANDIDATES:
        path = project_dir / relative_path
        if path.exists():
            return path

    module2_dir = project_dir / "models" / "module2"
    if module2_dir.exists():
        joblib_files = sorted(module2_dir.glob("*.joblib"))
        if joblib_files:
            return joblib_files[0]

    raise FileNotFoundError(
        "No Module 2 joblib model found. Tried tfidf_linearsvm.joblib, "
        "best_model.joblib, then any *.joblib under models/module2/."
    )


def get_model_classes(model: Any) -> list[str] | None:
    if hasattr(model, "classes_"):
        return [str(label) for label in model.classes_]

    named_steps = getattr(model, "named_steps", {})
    for step_name in ["linearsvc", "clf", "classifier", "svm", "estimator"]:
        step = named_steps.get(step_name)
        if step is not None and hasattr(step, "classes_"):
            return [str(label) for label in step.classes_]

    for step in getattr(named_steps, "values", lambda: [])():
        if hasattr(step, "classes_"):
            return [str(label) for label in step.classes_]

    return None


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def predict_sentiment(df: pd.DataFrame, text_column: str, model_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[prepare] Loading sentiment model: {model_path}")
    model = joblib.load(model_path)
    texts = df[text_column].fillna("").astype(str).str.strip()

    predictions = model.predict(texts)
    invalid_predictions = sorted(set(map(str, predictions)) - set(LABELS))
    if invalid_predictions:
        raise ValueError(f"Model predicted labels outside {LABELS}: {invalid_predictions}")

    df["predicted_sentiment"] = predictions
    df["sentiment_source"] = f"inferred:{model_path.name}"

    confidence = np.full(len(df), np.nan, dtype=float)
    margin = np.full(len(df), np.nan, dtype=float)
    has_decision_function = False

    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(texts))
            class_labels = get_model_classes(model)
            if scores.ndim == 1 and class_labels and len(class_labels) == 2:
                scores = np.column_stack([-scores, scores])
            elif scores.ndim == 1:
                scores = scores.reshape(-1, 1)

            if class_labels and scores.shape[1] == len(class_labels) and scores.shape[1] >= 2:
                sorted_scores = np.sort(scores, axis=1)
                margin = sorted_scores[:, -1] - sorted_scores[:, -2]
                confidence = np.max(stable_softmax(scores), axis=1)
                has_decision_function = True
            else:
                print("[prepare] Could not align decision scores with model classes.")
        except Exception as exc:  # pragma: no cover - defensive branch
            print(f"[prepare] Could not compute decision_function scores: {exc!r}")
    elif hasattr(model, "predict_proba"):
        try:
            probabilities = np.asarray(model.predict_proba(texts))
            confidence = np.max(probabilities, axis=1)
        except Exception as exc:  # pragma: no cover - defensive branch
            print(f"[prepare] Could not compute predict_proba confidence: {exc!r}")

    df["sentiment_confidence"] = confidence
    df["sentiment_margin"] = margin

    info = {
        "model_path": str(model_path),
        "has_decision_function": has_decision_function,
        "num_predictions": int(len(df)),
    }
    return df, info


def ensure_required_columns(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()

    if "doc_id" not in df.columns:
        df["doc_id"] = [f"doc_{index}" for index in range(len(df))]
    else:
        missing_doc_id = is_empty_text(df["doc_id"])
        if missing_doc_id.any():
            df.loc[missing_doc_id, "doc_id"] = [f"doc_{index}" for index in df.index[missing_doc_id]]

    if "comment" not in df.columns:
        df["comment"] = df[text_column]
    else:
        empty_comment = is_empty_text(df["comment"])
        df.loc[empty_comment, "comment"] = df.loc[empty_comment, text_column]

    if "retrieval_text" not in df.columns:
        df["retrieval_text"] = df["comment"]
    else:
        empty_retrieval = is_empty_text(df["retrieval_text"])
        df.loc[empty_retrieval, "retrieval_text"] = df.loc[empty_retrieval, "comment"]

    for column in ["category", "product_name", "product_id", "rating"]:
        if column not in df.columns:
            df[column] = np.nan

    for column in ["predicted_sentiment", "sentiment_confidence", "sentiment_margin", "sentiment_source"]:
        if column not in df.columns:
            df[column] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Module 4 RAG corpus inputs.")
    parser.add_argument("--project_dir", default=None, help="Project root. Defaults to this script's parent project.")
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    dirs = ensure_dirs(project_dir)
    output_path = dirs["processed"] / "rag_corpus_module4.csv"
    summary_path = dirs["results"] / "module4_prepare_summary.json"

    input_path = first_existing_input(project_dir)
    print(f"[prepare] Project dir: {project_dir}")
    print(f"[prepare] Input corpus: {input_path}")

    df = pd.read_csv(input_path)
    original_rows = int(len(df))
    text_column = choose_text_column(df)
    print(f"[prepare] Using text column for inference: {text_column}")

    df = ensure_required_columns(df, text_column)
    before_drop = len(df)
    non_empty_mask = ~is_empty_text(df["retrieval_text"]) & ~is_empty_text(df["comment"])
    df = df.loc[non_empty_mask].copy().reset_index(drop=True)
    dropped_empty_rows = int(before_drop - len(df))

    predicted_missing_mask = is_empty_text(df["predicted_sentiment"])
    sentiment_columns_exist = "predicted_sentiment" in df.columns and not predicted_missing_mask.any()

    inference_info: dict[str, Any] = {
        "sentiment_was_inferred": False,
        "model_path": None,
        "has_decision_function": False,
    }

    if sentiment_columns_exist:
        print("[prepare] Existing predicted_sentiment column found; preserving it.")
        df["sentiment_source"] = df["sentiment_source"].fillna("existing").replace("", "existing")
        for column in ["sentiment_confidence", "sentiment_margin"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    else:
        model_path = find_model_path(project_dir)
        if "predicted_sentiment" in df.columns and not predicted_missing_mask.all():
            print("[prepare] predicted_sentiment has missing rows; inferring only missing labels.")
            inferred_df, inference_info = predict_sentiment(df.copy(), text_column, model_path)
            fill_mask = predicted_missing_mask.reset_index(drop=True)
            for column in [
                "predicted_sentiment",
                "sentiment_confidence",
                "sentiment_margin",
                "sentiment_source",
            ]:
                df.loc[fill_mask, column] = inferred_df.loc[fill_mask, column].values
            df["sentiment_source"] = df["sentiment_source"].fillna("existing").replace("", "existing")
        else:
            df, inference_info = predict_sentiment(df, text_column, model_path)
        inference_info["sentiment_was_inferred"] = True

    invalid_sentiments = sorted(set(df["predicted_sentiment"].dropna().astype(str)) - set(LABELS))
    if invalid_sentiments:
        raise ValueError(f"Invalid predicted_sentiment values: {invalid_sentiments}")

    required_columns = [
        "doc_id",
        "retrieval_text",
        "comment",
        "category",
        "product_name",
        "product_id",
        "rating",
        "predicted_sentiment",
        "sentiment_confidence",
        "sentiment_margin",
        "sentiment_source",
    ]
    first_columns = [column for column in required_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in first_columns]
    df = df[first_columns + remaining_columns]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_rows": original_rows,
        "output_rows": int(len(df)),
        "dropped_empty_rows": dropped_empty_rows,
        "text_column_used": text_column,
        "sentiment_columns_preserved": bool(sentiment_columns_exist),
        **inference_info,
        "predicted_sentiment_distribution": {
            str(key): int(value) for key, value in df["predicted_sentiment"].value_counts(dropna=False).items()
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prepare] Saved prepared corpus: {output_path}")
    print(f"[prepare] Saved summary: {summary_path}")
    print(f"[prepare] Output rows: {len(df):,}")


if __name__ == "__main__":
    main()
