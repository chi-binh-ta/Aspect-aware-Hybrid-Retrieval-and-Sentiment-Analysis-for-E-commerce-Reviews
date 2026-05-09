# %% [markdown]
# # Module 2: Baseline Sentiment Classification
#
# Module này xây baseline sentiment classification cho dữ liệu:
#
# `/content/ecommerce-sentiment-rag/data/processed/sentiment_reviews.csv`
#
# Phạm vi:
#
# - Không dùng RAG corpus.
# - Không fine-tune transformer/PhoBERT.
# - Chỉ dùng baseline cổ điển với TF-IDF + scikit-learn.
# - Tuning nhẹ trên validation set, không dùng test set để chọn tham số.

# %% [markdown]
# ## 1. Cài đặt thư viện cần thiết

# %%
import importlib
import subprocess
import sys


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "joblib",
]


def ensure_packages(packages):
    missing = []
    import_names = {
        "scikit-learn": "sklearn",
        "joblib": "joblib",
    }
    for package in packages:
        import_name = import_names.get(package, package.replace("-", "_"))
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages(REQUIRED_PACKAGES)

# %% [markdown]
# ## 2. Imports và cấu hình

# %%
from pathlib import Path
import itertools
import json
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]
VALID_SPLITS = {"train", "valid", "test"}

def resolve_project_dir() -> Path:
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    if Path("/content").exists():
        return Path("/content/ecommerce-sentiment-rag")
    current = Path.cwd().resolve()
    if (current / "data").exists() and (current / "src").exists():
        return current
    try:
        script_path = Path(__file__).resolve()
        for candidate in [script_path.parent, *script_path.parents]:
            if candidate.name == "ecommerce-sentiment-rag":
                return candidate
    except NameError:
        pass
    if current.name == "ecommerce-sentiment-rag":
        return current
    return current / "ecommerce-sentiment-rag"


PROJECT_DIR = resolve_project_dir()

DATA_PATH = PROJECT_DIR / "data" / "processed" / "sentiment_reviews.csv"
MODEL_DIR = PROJECT_DIR / "models" / "module2"
RESULTS_DIR = PROJECT_DIR / "results" / "module2"
FIGURES_DIR = PROJECT_DIR / "figures" / "module2"

for directory in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

METRICS_SUMMARY_CSV = RESULTS_DIR / "metrics_summary.csv"
METRICS_SUMMARY_JSON = RESULTS_DIR / "metrics_summary.json"
TUNING_RESULTS_CSV = RESULTS_DIR / "tuning_results.csv"

LOGREG_REPORT_CSV = RESULTS_DIR / "logistic_regression_classification_report.csv"
SVM_REPORT_CSV = RESULTS_DIR / "linear_svm_classification_report.csv"

LOGREG_PREDICTIONS_CSV = RESULTS_DIR / "logistic_regression_predictions.csv"
SVM_PREDICTIONS_CSV = RESULTS_DIR / "linear_svm_predictions.csv"

LOGREG_ERRORS_CSV = RESULTS_DIR / "error_analysis_logistic_regression.csv"
SVM_ERRORS_CSV = RESULTS_DIR / "error_analysis_linear_svm.csv"

LOGREG_MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"
SVM_MODEL_PATH = MODEL_DIR / "tfidf_linearsvm.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"

print("Project directory:", PROJECT_DIR)
print("Input data:", DATA_PATH)

# %% [markdown]
# ## 3. Load dữ liệu và kiểm tra schema

# %%
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {DATA_PATH}. Hãy chạy Module 1 trước hoặc copy "
        "sentiment_reviews.csv vào data/processed/."
    )

df = pd.read_csv(DATA_PATH)
required_columns = {"review_id", "text", "sentiment_label", "split"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise KeyError(f"Thiếu cột bắt buộc: {sorted(missing_columns)}")

df["text"] = df["text"].fillna("").astype(str).str.strip()
empty_text_mask = df["text"].str.len() == 0
num_empty_text_removed = int(empty_text_mask.sum())
if num_empty_text_removed:
    df = df.loc[~empty_text_mask].copy()
print("Số dòng text rỗng bị bỏ:", num_empty_text_removed)

invalid_labels = sorted(set(df["sentiment_label"].dropna().unique()) - set(LABELS))
if invalid_labels:
    raise ValueError(f"sentiment_label có giá trị ngoài {LABELS}: {invalid_labels}")

invalid_splits = sorted(set(df["split"].dropna().unique()) - VALID_SPLITS)
if invalid_splits:
    raise ValueError(f"split có giá trị ngoài {sorted(VALID_SPLITS)}: {invalid_splits}")

for split in ["train", "valid", "test"]:
    if (df["split"] == split).sum() == 0:
        raise ValueError(f"Split '{split}' đang rỗng.")

print("Shape tổng thể:", df.shape)
print("\nPhân bố nhãn theo split:")
split_label_distribution = pd.crosstab(df["split"], df["sentiment_label"])
print(split_label_distribution)

# %% [markdown]
# ## 4. Tách train / validation / test

# %%
train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
valid_df = df[df["split"] == "valid"].copy().reset_index(drop=True)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

X_train = train_df["text"].astype(str)
y_train = train_df["sentiment_label"].astype(str)

X_valid = valid_df["text"].astype(str)
y_valid = valid_df["sentiment_label"].astype(str)

X_test = test_df["text"].astype(str)
y_test = test_df["sentiment_label"].astype(str)

print("Train:", train_df.shape)
print("Valid:", valid_df.shape)
print("Test :", test_df.shape)

# %% [markdown]
# ## 5. Hàm evaluation và lưu kết quả

# %%
def compute_metrics(y_true, y_pred, labels=LABELS):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for index, label in enumerate(labels):
        per_class[label] = {
            "precision": float(per_class_precision[index]),
            "recall": float(per_class_recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(per_class_support[index]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def metrics_row(model_name, split, metrics, params=None, selected_by=None):
    row = {
        "model_name": model_name,
        "split": split,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "selected_by": selected_by or "",
        "params": json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
    }
    for label in LABELS:
        row[f"{label}_precision"] = metrics["per_class"][label]["precision"]
        row[f"{label}_recall"] = metrics["per_class"][label]["recall"]
        row[f"{label}_f1"] = metrics["per_class"][label]["f1"]
        row[f"{label}_support"] = metrics["per_class"][label]["support"]
    return row


def classification_report_dataframe(y_true, y_pred, model_name, split):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {
                "model_name": model_name,
                "split": split,
                "label": label,
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1_score": values.get("f1-score"),
                "support": values.get("support"),
            }
        else:
            row = {
                "model_name": model_name,
                "split": split,
                "label": label,
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": values,
                "support": np.nan,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def make_predictions_dataframe(source_df, y_true, y_pred, split, model_name):
    output = pd.DataFrame(
        {
            "review_id": source_df["review_id"].values,
            "text": source_df["text"].values,
            "true_label": list(y_true),
            "predicted_label": list(y_pred),
            "correct": np.array(list(y_true)) == np.array(list(y_pred)),
            "split": split,
            "model_name": model_name,
        }
    )
    return output


def build_error_analysis(predictions_df, source_df):
    optional_columns = [
        column
        for column in ["text_length", "rating", "sentiment_5class"]
        if column in source_df.columns
    ]
    metadata = source_df[["review_id", *optional_columns]].copy()
    errors = predictions_df.loc[~predictions_df["correct"]].merge(
        metadata, on="review_id", how="left"
    )
    if "text_length" not in errors.columns:
        errors["text_length"] = errors["text"].astype(str).str.split().str.len()
    errors["neutral_error_priority"] = (
        (errors["true_label"] == "neutral") | (errors["predicted_label"] == "neutral")
    ).astype(int)
    errors = errors.sort_values(
        ["neutral_error_priority", "text_length"],
        ascending=[False, True],
    )
    keep_columns = [
        "review_id",
        "text",
        "true_label",
        "predicted_label",
        "text_length",
        *[column for column in ["rating", "sentiment_5class"] if column in errors.columns],
    ]
    return errors[keep_columns].reset_index(drop=True)


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_metric_comparison(metrics_df, metric, title, output_path):
    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    test_rows = test_rows.sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=test_rows, x="model_name", y=metric, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

# %% [markdown]
# ## 6. Majority-class baseline

# %%
majority_model = DummyClassifier(strategy="most_frequent")
majority_model.fit(X_train, y_train)

majority_class = majority_model.classes_[np.argmax(majority_model.class_prior_)]
valid_pred_majority = majority_model.predict(X_valid)
test_pred_majority = majority_model.predict(X_test)

majority_valid_metrics = compute_metrics(y_valid, valid_pred_majority)
majority_test_metrics = compute_metrics(y_test, test_pred_majority)

print("Majority class từ train set:", majority_class)
print("Validation macro-F1:", round(majority_valid_metrics["macro_f1"], 4))
print("Test macro-F1:", round(majority_test_metrics["macro_f1"], 4))

# %% [markdown]
# ## 7. Candidate configs cho tuning nhẹ trên validation set
#
# Không dùng cross-validation vì dữ liệu đã có split cố định.
# Mỗi candidate được fit trên train và evaluate trên validation.
# Test set không được dùng trong bước chọn tham số.

# %%
LOGREG_CANDIDATES = list(
    itertools.product(
        [(1, 1), (1, 2)],
        [1, 2, 3],
        [0.5, 1.0, 2.0],
        [None, "balanced"],
    )
)

SVM_CANDIDATES = list(
    itertools.product(
        [(1, 1), (1, 2)],
        [1, 2, 3],
        [0.5, 1.0, 2.0],
        [None, "balanced"],
    )
)

print("Logistic Regression candidates:", len(LOGREG_CANDIDATES))
print("LinearSVC candidates:", len(SVM_CANDIDATES))

# %% [markdown]
# ## 8. Fit trên train và chọn theo validation macro-F1

# %%
def make_tfidf_logreg_pipeline(ngram_range, min_df, C, class_weight):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    solver="lbfgs",
                    multi_class="auto",
                    C=C,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_tfidf_linearsvm_pipeline(ngram_range, min_df, C, class_weight):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_model_family(model_family, candidates, pipeline_factory):
    tuning_rows = []
    best_entry = None

    for candidate_index, (ngram_range, min_df, C, class_weight) in enumerate(candidates, start=1):
        params = {
            "ngram_range": list(ngram_range),
            "min_df": min_df,
            "C": C,
            "class_weight": class_weight,
        }
        candidate_id = f"{model_family}_{candidate_index:03d}"
        pipeline = pipeline_factory(ngram_range, min_df, C, class_weight)
        started_at = time.time()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", FutureWarning)
                pipeline.fit(X_train, y_train)
            valid_pred = pipeline.predict(X_valid)
            valid_metrics = compute_metrics(y_valid, valid_pred)
            fit_status = "ok"
            error_message = ""
        except Exception as exc:
            valid_metrics = {
                "accuracy": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "per_class": {
                    label: {
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "support": 0,
                    }
                    for label in LABELS
                },
                "confusion_matrix": None,
            }
            pipeline = None
            fit_status = "failed"
            error_message = repr(exc)

        elapsed = time.time() - started_at
        row = {
            "candidate_id": candidate_id,
            "model_family": model_family,
            "fit_status": fit_status,
            "error_message": error_message,
            "fit_time_sec": elapsed,
            **params,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_precision": valid_metrics["macro_precision"],
            "valid_macro_recall": valid_metrics["macro_recall"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "valid_confusion_matrix": json.dumps(
                valid_metrics["confusion_matrix"], ensure_ascii=False
            ),
        }
        tuning_rows.append(row)

        if fit_status == "ok":
            if best_entry is None or valid_metrics["macro_f1"] > best_entry["valid_metrics"]["macro_f1"]:
                best_entry = {
                    "candidate_id": candidate_id,
                    "model_family": model_family,
                    "pipeline": pipeline,
                    "params": params,
                    "valid_metrics": valid_metrics,
                    "valid_predictions": valid_pred,
                }

    if best_entry is None:
        raise RuntimeError(f"Tất cả candidate của {model_family} đều failed.")

    return best_entry, tuning_rows


best_logreg, tuning_rows_logreg = tune_model_family(
    "logistic_regression",
    LOGREG_CANDIDATES,
    make_tfidf_logreg_pipeline,
)

best_svm, tuning_rows_svm = tune_model_family(
    "linear_svm",
    SVM_CANDIDATES,
    make_tfidf_linearsvm_pipeline,
)

tuning_results = pd.DataFrame(tuning_rows_logreg + tuning_rows_svm)
tuning_results.to_csv(TUNING_RESULTS_CSV, index=False, encoding="utf-8-sig")

top10_tuning = tuning_results[tuning_results["fit_status"] == "ok"].sort_values(
    "valid_macro_f1", ascending=False
).head(10)

print("Top 10 cấu hình theo validation macro-F1:")
display_columns = [
    "candidate_id",
    "model_family",
    "ngram_range",
    "min_df",
    "C",
    "class_weight",
    "valid_accuracy",
    "valid_macro_f1",
    "valid_weighted_f1",
]
print(top10_tuning[display_columns].to_string(index=False))
print("Saved tuning results:", TUNING_RESULTS_CSV)

# %% [markdown]
# ## 9. Evaluate đúng một lần trên test sau khi chọn cấu hình

# %%
selected_models = {
    "logistic_regression": best_logreg,
    "linear_svm": best_svm,
}

for model_name, entry in selected_models.items():
    test_pred = entry["pipeline"].predict(X_test)
    entry["test_predictions"] = test_pred
    entry["test_metrics"] = compute_metrics(y_test, test_pred)

best_overall_name, best_overall_entry = sorted(
    selected_models.items(),
    key=lambda item: item[1]["valid_metrics"]["macro_f1"],
    reverse=True,
)[0]

print("Best Logistic Regression params:", best_logreg["params"])
print("Best Linear SVM params:", best_svm["params"])
print("Best overall model by validation macro-F1:", best_overall_name)

# %% [markdown]
# ## 10. Lưu metrics summary và classification reports

# %%
metrics_rows = [
    metrics_row(
        "majority_class",
        "valid",
        majority_valid_metrics,
        params={"strategy": "most_frequent", "majority_class": str(majority_class)},
    ),
    metrics_row(
        "majority_class",
        "test",
        majority_test_metrics,
        params={"strategy": "most_frequent", "majority_class": str(majority_class)},
    ),
]

for model_name, entry in selected_models.items():
    metrics_rows.append(
        metrics_row(
            model_name,
            "valid",
            entry["valid_metrics"],
            params=entry["params"],
            selected_by="valid_macro_f1",
        )
    )
    metrics_rows.append(
        metrics_row(
            model_name,
            "test",
            entry["test_metrics"],
            params=entry["params"],
            selected_by="valid_macro_f1",
        )
    )

metrics_summary = pd.DataFrame(metrics_rows)
metrics_summary.to_csv(METRICS_SUMMARY_CSV, index=False, encoding="utf-8-sig")

metrics_summary_json = {
    "selection_rule": "best model selected by validation macro-F1 only",
    "labels": LABELS,
    "majority_class": str(majority_class),
    "best_model_by_valid_macro_f1": best_overall_name,
    "best_model_candidate_id": best_overall_entry["candidate_id"],
    "best_model_params": best_overall_entry["params"],
    "metrics": metrics_rows,
    "confusion_matrices": {
        "majority_class": {
            "valid": majority_valid_metrics["confusion_matrix"],
            "test": majority_test_metrics["confusion_matrix"],
        },
        "logistic_regression": {
            "valid": best_logreg["valid_metrics"]["confusion_matrix"],
            "test": best_logreg["test_metrics"]["confusion_matrix"],
        },
        "linear_svm": {
            "valid": best_svm["valid_metrics"]["confusion_matrix"],
            "test": best_svm["test_metrics"]["confusion_matrix"],
        },
    },
}
METRICS_SUMMARY_JSON.write_text(
    json.dumps(metrics_summary_json, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

logreg_report = pd.concat(
    [
        classification_report_dataframe(
            y_valid,
            best_logreg["valid_predictions"],
            "logistic_regression",
            "valid",
        ),
        classification_report_dataframe(
            y_test,
            best_logreg["test_predictions"],
            "logistic_regression",
            "test",
        ),
    ],
    ignore_index=True,
)
logreg_report.to_csv(LOGREG_REPORT_CSV, index=False, encoding="utf-8-sig")

svm_report = pd.concat(
    [
        classification_report_dataframe(
            y_valid,
            best_svm["valid_predictions"],
            "linear_svm",
            "valid",
        ),
        classification_report_dataframe(
            y_test,
            best_svm["test_predictions"],
            "linear_svm",
            "test",
        ),
    ],
    ignore_index=True,
)
svm_report.to_csv(SVM_REPORT_CSV, index=False, encoding="utf-8-sig")

print("Saved metrics summary:", METRICS_SUMMARY_CSV)
print("Saved metrics JSON:", METRICS_SUMMARY_JSON)
print("Saved reports:", LOGREG_REPORT_CSV, SVM_REPORT_CSV)

# %% [markdown]
# ## 11. Lưu predictions và error analysis

# %%
logreg_predictions = pd.concat(
    [
        make_predictions_dataframe(
            valid_df,
            y_valid,
            best_logreg["valid_predictions"],
            "valid",
            "logistic_regression",
        ),
        make_predictions_dataframe(
            test_df,
            y_test,
            best_logreg["test_predictions"],
            "test",
            "logistic_regression",
        ),
    ],
    ignore_index=True,
)
logreg_predictions.to_csv(LOGREG_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

svm_predictions = pd.concat(
    [
        make_predictions_dataframe(
            valid_df,
            y_valid,
            best_svm["valid_predictions"],
            "valid",
            "linear_svm",
        ),
        make_predictions_dataframe(
            test_df,
            y_test,
            best_svm["test_predictions"],
            "test",
            "linear_svm",
        ),
    ],
    ignore_index=True,
)
svm_predictions.to_csv(SVM_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

logreg_test_predictions = logreg_predictions[logreg_predictions["split"] == "test"].copy()
svm_test_predictions = svm_predictions[svm_predictions["split"] == "test"].copy()

logreg_errors = build_error_analysis(logreg_test_predictions, test_df)
svm_errors = build_error_analysis(svm_test_predictions, test_df)

logreg_errors.to_csv(LOGREG_ERRORS_CSV, index=False, encoding="utf-8-sig")
svm_errors.to_csv(SVM_ERRORS_CSV, index=False, encoding="utf-8-sig")

print("Saved predictions:", LOGREG_PREDICTIONS_CSV, SVM_PREDICTIONS_CSV)
print("Saved error analysis:", LOGREG_ERRORS_CSV, SVM_ERRORS_CSV)

# %% [markdown]
# ## 12. Vẽ figures

# %%
sns.set_theme(style="whitegrid", font_scale=1.0)

plot_confusion_matrix(
    y_test,
    best_logreg["test_predictions"],
    "Logistic Regression - Test Confusion Matrix",
    FIGURES_DIR / "logistic_regression_confusion_matrix.png",
)

plot_confusion_matrix(
    y_test,
    best_svm["test_predictions"],
    "Linear SVM - Test Confusion Matrix",
    FIGURES_DIR / "linear_svm_confusion_matrix.png",
)

plot_metric_comparison(
    metrics_summary,
    "accuracy",
    "Test Accuracy Comparison",
    FIGURES_DIR / "model_comparison_accuracy.png",
)

plot_metric_comparison(
    metrics_summary,
    "macro_f1",
    "Test Macro-F1 Comparison",
    FIGURES_DIR / "model_comparison_macro_f1.png",
)

print("Saved figures:", FIGURES_DIR)

# %% [markdown]
# ## 13. Lưu models và label mapping

# %%
joblib.dump(best_logreg["pipeline"], LOGREG_MODEL_PATH)
joblib.dump(best_svm["pipeline"], SVM_MODEL_PATH)
joblib.dump(best_overall_entry["pipeline"], BEST_MODEL_PATH)

label_mapping = {
    "labels": LABELS,
    "label_to_id": {label: index for index, label in enumerate(LABELS)},
    "id_to_label": {str(index): label for index, label in enumerate(LABELS)},
    "best_model_by_valid_macro_f1": best_overall_name,
    "best_model_candidate_id": best_overall_entry["candidate_id"],
    "best_model_params": best_overall_entry["params"],
}

LABEL_MAPPING_PATH.write_text(
    json.dumps(label_mapping, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Saved model:", LOGREG_MODEL_PATH)
print("Saved model:", SVM_MODEL_PATH)
print("Saved best model:", BEST_MODEL_PATH)
print("Saved label mapping:", LABEL_MAPPING_PATH)

# %% [markdown]
# ## 14. Validation cuối

# %%
expected_output_files = [
    METRICS_SUMMARY_CSV,
    METRICS_SUMMARY_JSON,
    TUNING_RESULTS_CSV,
    LOGREG_REPORT_CSV,
    SVM_REPORT_CSV,
    LOGREG_PREDICTIONS_CSV,
    SVM_PREDICTIONS_CSV,
    LOGREG_ERRORS_CSV,
    SVM_ERRORS_CSV,
    FIGURES_DIR / "logistic_regression_confusion_matrix.png",
    FIGURES_DIR / "linear_svm_confusion_matrix.png",
    FIGURES_DIR / "model_comparison_accuracy.png",
    FIGURES_DIR / "model_comparison_macro_f1.png",
    LOGREG_MODEL_PATH,
    SVM_MODEL_PATH,
    BEST_MODEL_PATH,
    LABEL_MAPPING_PATH,
]

for path in expected_output_files:
    assert path.exists(), f"Missing output file: {path}"
    assert path.stat().st_size > 0, f"Empty output file: {path}"

metrics_check = pd.read_csv(METRICS_SUMMARY_CSV)
assert not metrics_check.empty, "metrics_summary is empty"

logreg_predictions_check = pd.read_csv(LOGREG_PREDICTIONS_CSV)
svm_predictions_check = pd.read_csv(SVM_PREDICTIONS_CSV)

expected_prediction_rows = len(valid_df) + len(test_df)
assert len(logreg_predictions_check) == expected_prediction_rows, (
    "Logistic Regression predictions row count mismatch"
)
assert len(svm_predictions_check) == expected_prediction_rows, (
    "Linear SVM predictions row count mismatch"
)

for predictions_check, name in [
    (logreg_predictions_check, "logistic_regression"),
    (svm_predictions_check, "linear_svm"),
]:
    assert set(predictions_check["predicted_label"].unique()).issubset(set(LABELS)), (
        f"{name} has predicted_label outside {LABELS}"
    )
    assert set(predictions_check["split"].unique()) == {"valid", "test"}, (
        f"{name} predictions must contain valid and test splits"
    )

assert LOGREG_MODEL_PATH.exists() and LOGREG_MODEL_PATH.stat().st_size > 0
assert SVM_MODEL_PATH.exists() and SVM_MODEL_PATH.stat().st_size > 0
assert BEST_MODEL_PATH.exists() and BEST_MODEL_PATH.stat().st_size > 0

print("All Module 2 validation checks passed.")

# %% [markdown]
# ## 15. Kết quả cuối

# %%
print("\n=== Metrics summary ===")
print(
    metrics_summary[
        [
            "model_name",
            "split",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ]
    ].to_string(index=False)
)

best_test_metrics = best_overall_entry["test_metrics"]
best_test_predictions = best_overall_entry["test_predictions"]

print("\n=== Best model by validation macro-F1 ===")
print("Model:", best_overall_name)
print("Candidate ID:", best_overall_entry["candidate_id"])
print("Params:", best_overall_entry["params"])
print("Validation macro-F1:", round(best_overall_entry["valid_metrics"]["macro_f1"], 6))
print("Test accuracy:", round(best_test_metrics["accuracy"], 6))
print("Test macro-F1:", round(best_test_metrics["macro_f1"], 6))

print("\n=== Classification report on test set for best model ===")
print(classification_report(y_test, best_test_predictions, labels=LABELS, zero_division=0))

print("\n=== Output files ===")
for path in expected_output_files:
    print(path)
