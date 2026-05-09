# %% [markdown]
# # Module 1: Data Layer cho Ecommerce Sentiment + RAG
#
# Notebook/script này chỉ thực hiện data layer:
#
# - Data loading
# - Cleaning
# - Schema normalization
# - Train/valid/test split cho sentiment dataset
# - RAG corpus construction
# - EDA figures
# - Data quality reports
#
# Không train model trong Module 1.

# %%
import importlib
import subprocess
import sys


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "openpyxl",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
]


def ensure_packages(packages):
    """Install missing packages in Colab or a fresh Python environment."""
    missing = []
    import_names = {
        "scikit-learn": "sklearn",
        "openpyxl": "openpyxl",
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
# ## 1. Imports và cấu hình đường dẫn

# %%
from collections import Counter
from pathlib import Path
import hashlib
import html
import json
import os
import re
import shutil
import unicodedata
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


RANDOM_STATE = 42

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

RAW_DIR = PROJECT_DIR / "data" / "raw"
RAG_JSON_DIR = RAW_DIR / "rag_json"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "data" / "reports"
FIGURES_DIR = PROJECT_DIR / "figures" / "module1"

for directory in [RAW_DIR, RAG_JSON_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_SENT_PATH = RAW_DIR / "Data_sent.xlsx"
RAG_ZIP_PATH = RAW_DIR / "RAG.zip"

SENTIMENT_REVIEWS_PATH = PROCESSED_DIR / "sentiment_reviews.csv"
SENTIMENT_REVIEWS_5CLASS_PATH = PROCESSED_DIR / "sentiment_reviews_5class.csv"
SENTIMENT_DUPLICATES_REPORT_PATH = PROCESSED_DIR / "sentiment_duplicates_report.csv"
RAG_DOCUMENTS_JSONL_PATH = PROCESSED_DIR / "rag_documents.jsonl"
RAG_CORPUS_CSV_PATH = PROCESSED_DIR / "rag_corpus.csv"

DATA_SUMMARY_PATH = REPORTS_DIR / "data_summary.json"
SENTIMENT_LABEL_DISTRIBUTION_PATH = REPORTS_DIR / "sentiment_label_distribution.csv"
RAG_CATEGORY_SPLIT_DISTRIBUTION_PATH = REPORTS_DIR / "rag_category_split_distribution.csv"

print("Project directory:", PROJECT_DIR)

# %% [markdown]
# ## 2. Tìm và copy input files
#
# Code sẽ tìm `Data_sent.xlsx` và `RAG.zip` tại:
#
# - `/content/Data_sent.xlsx`, `/content/RAG.zip`
# - Working directory hiện tại
# - Thư mục chứa notebook/script
# - `data/raw/` nếu đã copy trước đó
#
# Nếu chưa có file, upload vào `/content` trong Colab rồi chạy lại cell này.

# %%
def notebook_or_script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def limited_search(filename, roots):
    matches = []
    seen_roots = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        try:
            matches.extend(root.rglob(filename))
        except OSError:
            continue
    return matches


def find_input_file(filename):
    direct_candidates = [
        RAW_DIR / filename,
        Path("/content") / filename,
        Path.cwd() / filename,
        notebook_or_script_dir() / filename,
    ]

    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    search_roots = [Path("/content"), Path.cwd(), notebook_or_script_dir()]
    matches = limited_search(filename, search_roots)
    matches = [path for path in matches if path.is_file()]
    return matches[0] if matches else None


def copy_input_to_raw(filename, destination):
    source = find_input_file(filename)
    if source is None:
        print(f"\nKhông tìm thấy {filename}.")
        print("Cách upload trong Google Colab:")
        print("1. Mở panel Files bên trái.")
        print(f"2. Upload {filename} vào /content.")
        print("3. Chạy lại cell này.")
        print("\nHoặc chạy thủ công trong Colab:")
        print("from google.colab import files")
        print("files.upload()")
        raise FileNotFoundError(f"Missing required input file: {filename}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
        print(f"Copied {source} -> {destination}")
    else:
        print(f"Found {filename} at {destination}")

    return destination


DATA_SENT_PATH = copy_input_to_raw("Data_sent.xlsx", DATA_SENT_PATH)
RAG_ZIP_PATH = copy_input_to_raw("RAG.zip", RAG_ZIP_PATH)

# %% [markdown]
# ## 3. Hàm tiện ích cleaning và serialization

# %%
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
VALID_SENTIMENT_RATINGS = {1, 2, 3, 4, 5}
SENTIMENT_MAP = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def clean_text(value):
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_duplicate(text):
    text = clean_text(text).lower()
    return WHITESPACE_RE.sub(" ", text).strip()


def word_count(text):
    text = clean_text(text)
    if not text:
        return 0
    return len(text.split())


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def value_to_text(value):
    if is_missing(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return compact_json(value)
    return clean_text(value)


def sanitize_id(value):
    text = clean_text(value)
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", text)
    text = text.strip("_")
    return text or "missing"


def to_jsonable(value):
    if is_missing(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (dict, list, tuple, str, int, float, bool)):
        return value
    return str(value)


def read_json_records(path):
    raw_text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        records = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "records", "reviews", "items"]:
            if isinstance(data.get(key), list):
                return data[key]
        if data and all(isinstance(value, dict) for value in data.values()):
            return list(data.values())
        return [data]

    raise ValueError(f"Unsupported JSON structure in {path}")


def get_field(record, *candidate_keys):
    if not isinstance(record, dict):
        return None

    for key in candidate_keys:
        if key in record:
            return record[key]

    lower_map = {str(key).lower(): key for key in record.keys()}
    for key in candidate_keys:
        matched_key = lower_map.get(str(key).lower())
        if matched_key is not None:
            return record[matched_key]

    return None


def parse_rating(value):
    if is_missing(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    if np.isclose(numeric, round(numeric)):
        return int(round(numeric))
    return float(numeric)

# %% [markdown]
# ## 4. Xử lý `Data_sent.xlsx`

# %%
sentiment_raw = pd.read_excel(DATA_SENT_PATH)
sentiment_raw_rows = len(sentiment_raw)

sentiment_raw.columns = [clean_text(col) for col in sentiment_raw.columns]
lower_to_original_col = {col.lower(): col for col in sentiment_raw.columns}

text_col = lower_to_original_col.get("cmt") or lower_to_original_col.get("text")
rating_col = lower_to_original_col.get("sentiment") or lower_to_original_col.get("rating")

if text_col is None or rating_col is None:
    raise KeyError(
        "Không tìm thấy cột bắt buộc. File cần có cột 'Cmt' và 'sentiment' "
        f"hoặc cột tương đương text/rating. Columns hiện có: {list(sentiment_raw.columns)}"
    )

sentiment_df = sentiment_raw.rename(columns={text_col: "text", rating_col: "rating"})[
    ["text", "rating"]
].copy()

sentiment_df["text"] = sentiment_df["text"].apply(clean_text)
sentiment_df = sentiment_df[sentiment_df["text"].str.len() > 0].copy()

rating_numeric = pd.to_numeric(sentiment_df["rating"], errors="coerce")
sentiment_df = sentiment_df[rating_numeric.notna()].copy()
rating_numeric = pd.to_numeric(sentiment_df["rating"], errors="coerce")
sentiment_df = sentiment_df[np.isclose(rating_numeric, np.round(rating_numeric))].copy()
sentiment_df["rating"] = pd.to_numeric(sentiment_df["rating"], errors="coerce").round().astype(int)
sentiment_df = sentiment_df[sentiment_df["rating"].isin(VALID_SENTIMENT_RATINGS)].copy()

sentiment_df["sentiment_label"] = sentiment_df["rating"].map(SENTIMENT_MAP)
sentiment_df["sentiment_5class"] = "rating_" + sentiment_df["rating"].astype(str)
sentiment_df["text_length"] = sentiment_df["text"].apply(word_count)
sentiment_df["normalized_text"] = sentiment_df["text"].apply(normalize_for_duplicate)

sentiment_after_cleaning_rows = len(sentiment_df)

print("Sentiment raw rows:", sentiment_raw_rows)
print("Sentiment rows after cleaning:", sentiment_after_cleaning_rows)
sentiment_df.head()

# %% [markdown]
# ## 5. Kiểm tra và xử lý duplicate trong sentiment dataset
#
# Rule:
#
# - Duplicate cùng rating/sentiment: giữ 1 dòng.
# - Duplicate conflict rating hoặc conflict `sentiment_label`: loại khỏi tập chính để giảm label noise.

# %%
duplicate_report_columns = [
    "normalized_text",
    "num_occurrences",
    "unique_ratings",
    "unique_sentiment_labels",
    "conflict_label",
]

if sentiment_df.empty:
    raise ValueError("Sentiment dataset rỗng sau cleaning.")

duplicate_groups = (
    sentiment_df.groupby("normalized_text")
    .agg(
        num_occurrences=("normalized_text", "size"),
        unique_ratings=("rating", lambda values: compact_json(sorted(set(map(int, values))))),
        unique_sentiment_labels=(
            "sentiment_label",
            lambda values: compact_json(sorted(set(map(str, values)))),
        ),
        rating_nunique=("rating", "nunique"),
        sentiment_label_nunique=("sentiment_label", "nunique"),
    )
    .reset_index()
)

duplicate_report = duplicate_groups[duplicate_groups["num_occurrences"] > 1].copy()
duplicate_report["conflict_label"] = (
    (duplicate_report["rating_nunique"] > 1)
    | (duplicate_report["sentiment_label_nunique"] > 1)
)
duplicate_report = duplicate_report[duplicate_report_columns]
duplicate_report.to_csv(SENTIMENT_DUPLICATES_REPORT_PATH, index=False, encoding="utf-8-sig")

conflict_texts = set(duplicate_report.loc[duplicate_report["conflict_label"], "normalized_text"])
sentiment_removed_conflict_duplicates = int(
    sentiment_df["normalized_text"].isin(conflict_texts).sum()
)

sentiment_dedup = sentiment_df[~sentiment_df["normalized_text"].isin(conflict_texts)].copy()
sentiment_dedup = sentiment_dedup.drop_duplicates(subset=["normalized_text"], keep="first")
sentiment_dedup = sentiment_dedup.reset_index(drop=True)
sentiment_dedup.insert(
    0,
    "review_id",
    [f"sent_{index:06d}" for index in range(1, len(sentiment_dedup) + 1)],
)

sentiment_after_dedup_rows = len(sentiment_dedup)

print("Duplicate groups:", len(duplicate_report))
print("Rows removed because of conflict duplicates:", sentiment_removed_conflict_duplicates)
print("Sentiment rows after dedup:", sentiment_after_dedup_rows)
print("Duplicate report:", SENTIMENT_DUPLICATES_REPORT_PATH)

# %% [markdown]
# ## 6. Chia train/valid/test cho sentiment dataset
#
# Tỉ lệ mục tiêu: 70/15/15, stratify theo `sentiment_label`.
# Nếu dataset quá nhỏ để stratify an toàn, code sẽ cảnh báo và fallback sang random split.

# %%
def split_sentiment_dataset(df, label_col="sentiment_label", random_state=RANDOM_STATE):
    if df.empty:
        raise ValueError("Không thể split sentiment dataset rỗng.")

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=random_state,
            stratify=df[label_col],
        )
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=random_state,
            stratify=temp_df[label_col],
        )
    except ValueError as exc:
        warnings.warn(
            "Stratified split không thực hiện được, fallback sang random split. "
            f"Lý do: {exc}",
            RuntimeWarning,
        )
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=random_state,
            stratify=None,
        )
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=random_state,
            stratify=None,
        )

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    valid_df["split"] = "valid"
    test_df["split"] = "test"

    output = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    return output.sort_values("review_id").reset_index(drop=True)


sentiment_split = split_sentiment_dataset(sentiment_dedup)

sentiment_reviews = sentiment_split[
    ["review_id", "text", "rating", "sentiment_label", "text_length", "split"]
].copy()

sentiment_reviews_5class = sentiment_split[
    [
        "review_id",
        "text",
        "rating",
        "sentiment_label",
        "sentiment_5class",
        "text_length",
        "split",
    ]
].copy()

sentiment_reviews.to_csv(SENTIMENT_REVIEWS_PATH, index=False, encoding="utf-8-sig")
sentiment_reviews_5class.to_csv(SENTIMENT_REVIEWS_5CLASS_PATH, index=False, encoding="utf-8-sig")

print("Saved:", SENTIMENT_REVIEWS_PATH)
print("Saved:", SENTIMENT_REVIEWS_5CLASS_PATH)
print(sentiment_reviews["split"].value_counts())

# %% [markdown]
# ## 7. Xử lý `RAG.zip` và tạo RAG documents
#
# `retrieval_text` không đưa field `Response` vào vì đây có thể là phân tích sinh sẵn.

# %%
with zipfile.ZipFile(RAG_ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(RAG_JSON_DIR)

rag_json_files = sorted(path for path in RAG_JSON_DIR.rglob("*.json") if path.is_file())
if not rag_json_files:
    raise FileNotFoundError(f"Không tìm thấy file JSON nào sau khi giải nén vào {RAG_JSON_DIR}")

print("Number of RAG JSON files:", len(rag_json_files))
for path in rag_json_files[:5]:
    print("-", path.relative_to(RAG_JSON_DIR))

# %%
RAG_FILENAME_RE = re.compile(r"^(?P<category>.+?)-(?P<split>train|dev|test)\.json$", re.IGNORECASE)


def parse_category_split(path):
    match = RAG_FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(
            f"Không parse được category/split từ filename: {path.name}. "
            "Format mong đợi: Category-train.json, Category-dev.json, Category-test.json"
        )
    return match.group("category"), match.group("split").lower()


def build_retrieval_text(category, sub_category, product_name, rating, comment, detail_rating):
    parts = [
        f"Danh mục: {clean_text(category)}.",
        f"Loại sản phẩm: {value_to_text(sub_category)}.",
        f"Tên sản phẩm: {value_to_text(product_name)}.",
        f"Rating: {value_to_text(rating)}.",
        f"Nội dung đánh giá: {clean_text(comment)}.",
        f"Đánh giá chi tiết: {value_to_text(detail_rating)}.",
    ]
    return WHITESPACE_RE.sub(" ", " ".join(parts)).strip()


def make_doc_id(category, comment_id, product_id, comment):
    safe_category = sanitize_id(category)
    clean_comment_id = clean_text(comment_id)
    if clean_comment_id:
        return f"rag_{safe_category}_{sanitize_id(clean_comment_id)}"

    hash_input = f"{category}|{value_to_text(product_id)}|{clean_text(comment)}".encode("utf-8")
    digest = hashlib.sha1(hash_input).hexdigest()[:16]
    return f"rag_{safe_category}_{digest}"


rag_records = []
rag_num_empty_comments_removed = 0
rag_response_leak_violations = 0

for json_path in tqdm(rag_json_files, desc="Processing RAG JSON files"):
    category, split = parse_category_split(json_path)
    source_file = str(json_path.relative_to(RAG_JSON_DIR)).replace("\\", "/")
    records = read_json_records(json_path)

    for record_index, record in enumerate(records):
        comment = clean_text(get_field(record, "Comment"))
        if not comment:
            rag_num_empty_comments_removed += 1
            continue

        rating = parse_rating(get_field(record, "Rating"))
        product_name = get_field(record, "ProductName")
        product_id = get_field(record, "ProductId")
        comment_id = get_field(record, "CommentId")
        product_url = get_field(record, "ProductUrl")
        sub_category = get_field(record, "SubCategory")
        helpfulness_score = get_field(record, "Helpfulness_Score")
        detail_rating = get_field(record, "DetailRating")
        key_aspects = get_field(record, "KeyAspects")
        decision_making_advice = get_field(record, "DecisionMakingAdvice")
        image_helpfulness = get_field(record, "ImageHelpfulness")
        response = get_field(record, "Response")
        response_text = clean_text(response)
        has_response = bool(response_text)

        retrieval_text = build_retrieval_text(
            category=category,
            sub_category=sub_category,
            product_name=product_name,
            rating=rating,
            comment=comment,
            detail_rating=detail_rating,
        )

        # Short generic responses can accidentally appear inside comments.
        # This check catches real Response leakage without failing on tiny strings.
        if response_text and len(response_text) >= 20 and response_text in retrieval_text:
            rag_response_leak_violations += 1

        doc_id = make_doc_id(category, comment_id, product_id, comment)
        clean_comment_id = clean_text(comment_id)
        if clean_comment_id:
            dedup_key = f"comment_id::{category}::{clean_comment_id}"
        else:
            dedup_key = f"text_product::{value_to_text(product_id)}::{normalize_for_duplicate(comment)}"

        rag_records.append(
            {
                "doc_id": doc_id,
                "retrieval_text": retrieval_text,
                "comment": comment,
                "text_length": word_count(comment),
                "category": category,
                "split": split,
                "rating": rating,
                "product_name": value_to_text(product_name),
                "product_id": value_to_text(product_id),
                "comment_id": clean_comment_id,
                "product_url": value_to_text(product_url),
                "sub_category": value_to_text(sub_category),
                "helpfulness_score": value_to_text(helpfulness_score),
                "detail_rating": value_to_text(detail_rating),
                "key_aspects": to_jsonable(key_aspects),
                "decision_making_advice": to_jsonable(decision_making_advice),
                "image_helpfulness": to_jsonable(image_helpfulness),
                "source_file": source_file,
                "has_response": has_response,
                "dedup_key": dedup_key,
                "normalized_text": normalize_for_duplicate(comment),
            }
        )

rag_all = pd.DataFrame(rag_records)
if rag_all.empty:
    raise ValueError("RAG corpus rỗng sau khi loại comment rỗng.")

rag_before_dedup_rows = len(rag_all)
rag_corpus = rag_all.drop_duplicates(subset=["dedup_key"], keep="first").copy()

doc_id_duplicates_after_key_dedup = int(rag_corpus.duplicated(subset=["doc_id"]).sum())
if doc_id_duplicates_after_key_dedup:
    rag_corpus = rag_corpus.drop_duplicates(subset=["doc_id"], keep="first").copy()

rag_corpus = rag_corpus.reset_index(drop=True)
rag_num_duplicates_removed = rag_before_dedup_rows - len(rag_corpus)

print("RAG rows before dedup:", rag_before_dedup_rows)
print("Empty comments removed:", rag_num_empty_comments_removed)
print("RAG duplicates removed:", rag_num_duplicates_removed)
print("Response leakage candidates in retrieval_text:", rag_response_leak_violations)

# %% [markdown]
# ## 8. Lưu `rag_documents.jsonl` và `rag_corpus.csv`

# %%
rag_document_columns = [
    "doc_id",
    "retrieval_text",
    "category",
    "split",
    "rating",
    "product_name",
    "product_id",
    "comment_id",
    "product_url",
    "sub_category",
    "helpfulness_score",
    "detail_rating",
    "key_aspects",
    "decision_making_advice",
    "image_helpfulness",
    "source_file",
    "has_response",
]

rag_csv_columns = [
    "doc_id",
    "retrieval_text",
    "comment",
    "text_length",
    "category",
    "split",
    "rating",
    "product_name",
    "product_id",
    "comment_id",
    "product_url",
    "sub_category",
    "helpfulness_score",
    "detail_rating",
    "key_aspects",
    "decision_making_advice",
    "image_helpfulness",
    "source_file",
    "has_response",
]

with RAG_DOCUMENTS_JSONL_PATH.open("w", encoding="utf-8") as file:
    for _, row in rag_corpus.iterrows():
        metadata = {
            "category": row["category"],
            "split": row["split"],
            "rating": row["rating"],
            "product_name": row["product_name"],
            "product_id": row["product_id"],
            "comment_id": row["comment_id"],
            "product_url": row["product_url"],
            "sub_category": row["sub_category"],
            "helpfulness_score": row["helpfulness_score"],
            "detail_rating": row["detail_rating"],
            "key_aspects": row["key_aspects"],
            "decision_making_advice": row["decision_making_advice"],
            "image_helpfulness": row["image_helpfulness"],
            "source_file": row["source_file"],
            "has_response": bool(row["has_response"]),
        }
        document = {
            "doc_id": row["doc_id"],
            "retrieval_text": row["retrieval_text"],
            "metadata": metadata,
        }
        file.write(json.dumps(document, ensure_ascii=False, default=str) + "\n")

rag_corpus_for_csv = rag_corpus[rag_csv_columns].copy()
for column in ["key_aspects", "decision_making_advice", "image_helpfulness"]:
    rag_corpus_for_csv[column] = rag_corpus_for_csv[column].apply(
        lambda value: compact_json(value) if isinstance(value, (dict, list, tuple)) else value
    )

rag_corpus_for_csv.to_csv(RAG_CORPUS_CSV_PATH, index=False, encoding="utf-8-sig")

print("Saved:", RAG_DOCUMENTS_JSONL_PATH)
print("Saved:", RAG_CORPUS_CSV_PATH)

# %% [markdown]
# ## 9. Reports

# %%
def value_counts_dict(series, sort_index=True):
    counts = series.value_counts(dropna=False)
    if sort_index:
        try:
            counts = counts.sort_index()
        except TypeError:
            pass
    return {str(key): int(value) for key, value in counts.items()}


sentiment_label_distribution = (
    sentiment_reviews["sentiment_label"]
    .value_counts()
    .rename_axis("sentiment_label")
    .reset_index(name="count")
)
sentiment_label_distribution["percent"] = (
    sentiment_label_distribution["count"] / sentiment_label_distribution["count"].sum()
).round(6)
sentiment_label_distribution.to_csv(
    SENTIMENT_LABEL_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig"
)

rag_category_split_distribution = pd.crosstab(
    rag_corpus["category"], rag_corpus["split"]
).reset_index()
rag_category_split_distribution.to_csv(
    RAG_CATEGORY_SPLIT_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig"
)

sentiment_split_distribution = pd.crosstab(
    sentiment_reviews["split"], sentiment_reviews["sentiment_label"]
).to_dict(orient="index")
sentiment_split_distribution = {
    split: {label: int(count) for label, count in label_counts.items()}
    for split, label_counts in sentiment_split_distribution.items()
}

rag_split_distribution = value_counts_dict(rag_corpus["split"])
rag_rating_distribution = value_counts_dict(rag_corpus["rating"].dropna())

data_summary = {
    "sentiment_raw_rows": int(sentiment_raw_rows),
    "sentiment_after_cleaning_rows": int(sentiment_after_cleaning_rows),
    "sentiment_after_dedup_rows": int(sentiment_after_dedup_rows),
    "sentiment_removed_conflict_duplicates": int(sentiment_removed_conflict_duplicates),
    "sentiment_3class_distribution": value_counts_dict(
        sentiment_reviews["sentiment_label"], sort_index=False
    ),
    "sentiment_5class_distribution": value_counts_dict(
        sentiment_reviews_5class["sentiment_5class"]
    ),
    "sentiment_split_distribution": sentiment_split_distribution,
    "sentiment_avg_text_length": float(sentiment_reviews["text_length"].mean()),
    "rag_total_documents": int(len(rag_corpus)),
    "rag_category_distribution": value_counts_dict(rag_corpus["category"], sort_index=False),
    "rag_split_distribution": rag_split_distribution,
    "rag_rating_distribution": rag_rating_distribution,
    "rag_avg_text_length": float(rag_corpus["text_length"].mean()),
    "rag_num_empty_comments_removed": int(rag_num_empty_comments_removed),
    "rag_num_duplicates_removed": int(rag_num_duplicates_removed),
}

DATA_SUMMARY_PATH.write_text(
    json.dumps(data_summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("Saved:", DATA_SUMMARY_PATH)
print("Saved:", SENTIMENT_LABEL_DISTRIBUTION_PATH)
print("Saved:", RAG_CATEGORY_SPLIT_DISTRIBUTION_PATH)

# %% [markdown]
# ## 10. EDA figures

# %%
sns.set_theme(style="whitegrid", font_scale=1.0)


def save_countplot(data, x, order, title, xlabel, ylabel, output_path, rotation=0):
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=data, x=x, order=order, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


save_countplot(
    data=sentiment_reviews,
    x="sentiment_label",
    order=["negative", "neutral", "positive"],
    title="Sentiment 3-class Distribution",
    xlabel="Sentiment label",
    ylabel="Count",
    output_path=FIGURES_DIR / "sentiment_3class_distribution.png",
)

save_countplot(
    data=sentiment_reviews_5class,
    x="sentiment_5class",
    order=[f"rating_{rating}" for rating in range(1, 6)],
    title="Sentiment 5-class Distribution",
    xlabel="Rating class",
    ylabel="Count",
    output_path=FIGURES_DIR / "sentiment_5class_distribution.png",
)

plt.figure(figsize=(9, 5))
length_p99 = sentiment_reviews["text_length"].quantile(0.99)
sns.histplot(sentiment_reviews["text_length"].clip(upper=length_p99), bins=50, color="#59A14F")
plt.title("Sentiment Text Length Distribution")
plt.xlabel("Text length in words, clipped at P99")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sentiment_text_length_distribution.png", dpi=160)
plt.close()

category_order = rag_corpus["category"].value_counts().index.tolist()
save_countplot(
    data=rag_corpus,
    x="category",
    order=category_order,
    title="RAG Category Distribution",
    xlabel="Category",
    ylabel="Count",
    output_path=FIGURES_DIR / "rag_category_distribution.png",
    rotation=20,
)

rag_rating_order = sorted([rating for rating in rag_corpus["rating"].dropna().unique()])
save_countplot(
    data=rag_corpus,
    x="rating",
    order=rag_rating_order,
    title="RAG Rating Distribution",
    xlabel="Rating",
    ylabel="Count",
    output_path=FIGURES_DIR / "rag_rating_distribution.png",
)

print("Saved figures to:", FIGURES_DIR)

# %% [markdown]
# ## 11. Validation / Assertions

# %%
assert SENTIMENT_REVIEWS_PATH.exists(), f"Missing {SENTIMENT_REVIEWS_PATH}"
assert RAG_DOCUMENTS_JSONL_PATH.exists(), f"Missing {RAG_DOCUMENTS_JSONL_PATH}"
assert SENTIMENT_REVIEWS_PATH.stat().st_size > 0, f"Empty {SENTIMENT_REVIEWS_PATH}"
assert RAG_DOCUMENTS_JSONL_PATH.stat().st_size > 0, f"Empty {RAG_DOCUMENTS_JSONL_PATH}"

sentiment_check = pd.read_csv(SENTIMENT_REVIEWS_PATH)
rag_corpus_check = pd.read_csv(RAG_CORPUS_CSV_PATH)
rag_documents_check = pd.read_json(RAG_DOCUMENTS_JSONL_PATH, lines=True)

assert not sentiment_check.empty, "sentiment_reviews.csv is empty"
assert not rag_documents_check.empty, "rag_documents.jsonl is empty"

assert set(sentiment_check["sentiment_label"].unique()).issubset(
    {"negative", "neutral", "positive"}
), "Invalid sentiment_label values"

assert set(sentiment_check["rating"].dropna().astype(int).unique()).issubset(
    VALID_SENTIMENT_RATINGS
), "Invalid sentiment rating values"

assert set(sentiment_check["split"].unique()).issubset(
    {"train", "valid", "test"}
), "Invalid sentiment split values"

rag_splits_from_jsonl = {
    document["split"]
    for document in rag_documents_check["metadata"].tolist()
    if isinstance(document, dict)
}
assert rag_splits_from_jsonl.issubset({"train", "dev", "test"}), "Invalid RAG split values"

assert rag_documents_check["doc_id"].is_unique, "doc_id in rag_documents.jsonl must be unique"
assert rag_response_leak_violations == 0, "Response field leaked into retrieval_text"

assert sentiment_check["text"].fillna("").map(clean_text).str.len().gt(0).all(), (
    "Sentiment text contains empty values"
)
assert rag_corpus_check["comment"].fillna("").map(clean_text).str.len().gt(0).all(), (
    "RAG comment contains empty values"
)
assert rag_documents_check["retrieval_text"].fillna("").map(clean_text).str.len().gt(0).all(), (
    "RAG retrieval_text contains empty values"
)

print("All validation checks passed.")

# %% [markdown]
# ## 12. Output summary

# %%
print("\n=== Sentiment dataset ===")
print("Shape:", sentiment_check.shape)
print("\nSentiment label distribution:")
print(sentiment_check["sentiment_label"].value_counts())
print("\nRating 1-5 distribution:")
print(sentiment_check["rating"].value_counts().sort_index())

print("\n=== RAG corpus ===")
print("Shape:", rag_corpus_check.shape)
print("\nRAG category distribution:")
print(rag_corpus_check["category"].value_counts())
print("\nRAG split distribution:")
print(rag_corpus_check["split"].value_counts().sort_index())

print("\n=== Output files ===")
output_paths = [
    SENTIMENT_REVIEWS_PATH,
    SENTIMENT_REVIEWS_5CLASS_PATH,
    SENTIMENT_DUPLICATES_REPORT_PATH,
    RAG_DOCUMENTS_JSONL_PATH,
    RAG_CORPUS_CSV_PATH,
    DATA_SUMMARY_PATH,
    SENTIMENT_LABEL_DISTRIBUTION_PATH,
    RAG_CATEGORY_SPLIT_DISTRIBUTION_PATH,
    FIGURES_DIR / "sentiment_3class_distribution.png",
    FIGURES_DIR / "sentiment_5class_distribution.png",
    FIGURES_DIR / "sentiment_text_length_distribution.png",
    FIGURES_DIR / "rag_category_distribution.png",
    FIGURES_DIR / "rag_rating_distribution.png",
]

for path in output_paths:
    print(path)
