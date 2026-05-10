"""Product API for the Vietnamese e-commerce review intelligence dashboard."""

from __future__ import annotations

import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CORPUS_CANDIDATES = [
    PROJECT_DIR / "data" / "processed" / "rag_corpus_module4.csv",
    PROJECT_DIR / "data" / "processed" / "rag_corpus_with_sentiment.csv",
    PROJECT_DIR / "data" / "processed" / "rag_corpus.csv",
]

TEXT_COLUMNS = ["comment", "text", "clean_text", "review", "retrieval_text"]
SENTIMENT_COLUMNS = ["predicted_sentiment", "sentiment_label", "sentiment"]

ISSUE_TAXONOMY: dict[str, dict[str, Any]] = {
    "delivery": {
        "label_vi": "Giao hàng / vận chuyển",
        "keywords": ["giao hàng", "ship", "vận chuyển", "chậm", "lâu", "giao chậm", "giao lâu"],
        "action": "Rà soát đơn vị vận chuyển, SLA giao hàng và thông báo tracking cho khách.",
    },
    "packaging": {
        "label_vi": "Đóng gói / bao bì",
        "keywords": ["đóng gói", "móp", "rách", "hộp", "vỡ", "bẹp", "méo", "chống sốc"],
        "action": "Gia cố đóng gói, thêm chống sốc và kiểm tra tình trạng hộp trước khi gửi.",
    },
    "product_quality": {
        "label_vi": "Chất lượng sản phẩm",
        "keywords": ["lỗi", "hỏng", "kém", "không dùng được", "chất lượng", "bị lỗi", "bị hỏng"],
        "action": "Kiểm tra QC, nhà cung cấp và batch sản phẩm có tỷ lệ phàn nàn cao.",
    },
    "wrong_item": {
        "label_vi": "Sai mẫu / sai mô tả",
        "keywords": ["sai mẫu", "sai màu", "không giống", "khác hình", "khác mô tả", "sai size"],
        "action": "Cập nhật mô tả, ảnh, phân loại hàng và bước đối soát biến thể trước khi đóng gói.",
    },
    "missing_parts": {
        "label_vi": "Thiếu hàng / thiếu phụ kiện",
        "keywords": ["thiếu", "phụ kiện", "không đủ", "thiếu hàng", "thiếu đồ"],
        "action": "Thêm checklist trước khi giao và quy trình xử lý thiếu phụ kiện nhanh.",
    },
    "price_value": {
        "label_vi": "Giá / giá trị",
        "keywords": ["giá", "đắt", "rẻ", "đáng tiền", "tiền"],
        "action": "Theo dõi phản hồi về giá trị cảm nhận và điều chỉnh combo/khuyến mãi nếu cần.",
    },
    "service": {
        "label_vi": "Dịch vụ shop / hỗ trợ",
        "keywords": ["tư vấn", "phản hồi", "hỗ trợ", "không trả lời", "không rep"],
        "action": "Cải thiện thời gian phản hồi và kịch bản chăm sóc khách hàng sau mua.",
    },
}

STOPWORDS = {
    "khach", "khách", "hang", "hàng", "gi", "gì", "ve", "về", "co", "có",
    "khong", "không", "nhieu", "nhiều", "nhat", "nhất", "review", "san", "sản",
    "pham", "phẩm", "noi", "nói", "thuong", "thường",
}


class RagRequest(BaseModel):
    query: str | None = None
    question: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    sentiment: str | None = None
    category: str | None = None
    rating_min: float | None = None
    rating_max: float | None = None
    product: str | None = None
    product_name_contains: str | None = None
    use_dense: bool = True

    class Config:
        extra = "allow"


app = FastAPI(
    title="Vietnamese E-commerce Review Intelligence API",
    version="1.0.0",
    description="Product-style analytics and evidence-based RAG assistant over Vietnamese e-commerce reviews.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _first_existing_corpus() -> Path:
    for path in CORPUS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No processed RAG corpus found under data/processed.")


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    path = _first_existing_corpus()
    df = pd.read_csv(path)
    df = df.copy()
    df["_source_corpus_path"] = str(path.relative_to(PROJECT_DIR))
    text_col = get_text_column(df)
    df["_review_text"] = df[text_col].fillna("").astype(str)
    sentiment_col = get_sentiment_column(df)
    df["_sentiment"] = (
        df[sentiment_col].fillna("unknown").astype(str).str.lower()
        if sentiment_col
        else "unknown"
    )
    if "rating" in df.columns:
        df["_rating_numeric"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["_rating_numeric"] = math.nan
    return df


def get_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMNS:
        if column in df.columns:
            return column
    raise KeyError(f"No review text column found. Tried: {TEXT_COLUMNS}")


def get_sentiment_column(df: pd.DataFrame) -> str | None:
    for column in SENTIMENT_COLUMNS:
        if column in df.columns:
            return column
    return None


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def contains_any(text: str, keywords: list[str]) -> bool:
    lowered = normalize_text(text)
    return any(normalize_text(keyword) in lowered for keyword in keywords)


def query_tokens(query: str) -> set[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", normalize_text(query))
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


def apply_common_filters(
    df: pd.DataFrame,
    sentiment: str | None = None,
    category: str | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
    product: str | None = None,
) -> pd.DataFrame:
    filtered = df
    if sentiment:
        filtered = filtered[filtered["_sentiment"] == normalize_text(sentiment)]
    if category and "category" in filtered.columns:
        filtered = filtered[filtered["category"].fillna("").astype(str).map(normalize_text) == normalize_text(category)]
    if rating_min is not None:
        filtered = filtered[filtered["_rating_numeric"] >= rating_min]
    if rating_max is not None:
        filtered = filtered[filtered["_rating_numeric"] <= rating_max]
    if product:
        product_norm = normalize_text(product)
        product_name = filtered.get("product_name", pd.Series("", index=filtered.index)).fillna("").astype(str)
        product_id = filtered.get("product_id", pd.Series("", index=filtered.index)).fillna("").astype(str)
        filtered = filtered[
            product_name.map(normalize_text).str.contains(product_norm, regex=False)
            | product_id.map(normalize_text).str.contains(product_norm, regex=False)
        ]
    return filtered


def review_item(row: pd.Series, match_reason: str | None = None, score: float | None = None) -> dict[str, Any]:
    rating = row.get("_rating_numeric")
    rating_value = None if pd.isna(rating) else float(rating)
    return {
        "id": str(row.get("doc_id") or row.get("review_id") or row.name),
        "review_text": str(row.get("_review_text", "")),
        "sentiment": str(row.get("_sentiment", "unknown")),
        "rating": rating_value,
        "category": None if pd.isna(row.get("category", None)) else row.get("category", None),
        "product_name": None if pd.isna(row.get("product_name", None)) else row.get("product_name", None),
        "product_id": None if pd.isna(row.get("product_id", None)) else str(row.get("product_id")),
        "source": row.get("_source_corpus_path", "data/processed"),
        "score_or_match_reason": match_reason or (f"score={score:.4f}" if score is not None else "metadata match"),
    }


def keyword_search(
    query: str,
    top_k: int = 5,
    sentiment: str | None = None,
    category: str | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
    product: str | None = None,
) -> list[dict[str, Any]]:
    df = apply_common_filters(load_reviews(), sentiment, category, rating_min, rating_max, product)
    terms = query_tokens(query)
    if terms:
        scores = df["_review_text"].fillna("").astype(str).map(
            lambda text: sum(1 for token in terms if token in normalize_text(text))
        )
        df = df.assign(_keyword_score=scores)
        df = df[df["_keyword_score"] > 0].sort_values("_keyword_score", ascending=False)
    else:
        df = df.head(top_k).assign(_keyword_score=0)
    return [
        review_item(row, match_reason="keyword fallback", score=float(row.get("_keyword_score", 0)))
        for _, row in df.head(top_k).iterrows()
    ]


@lru_cache(maxsize=1)
def get_dense_retriever() -> Any:
    from module4_retrieve import ReviewRetriever

    return ReviewRetriever(project_dir=PROJECT_DIR)


def dense_retrieve(request: RagRequest, query: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not request.use_dense:
        return keyword_search(
            query,
            top_k=request.top_k,
            sentiment=request.sentiment,
            category=request.category,
            rating_min=request.rating_min,
            rating_max=request.rating_max,
            product=request.product or request.product_name_contains,
        ), ["Dense retrieval skipped by request; keyword fallback used."]
    try:
        retriever = get_dense_retriever()
        result = retriever.retrieve(
            query=query,
            top_k=request.top_k,
            category=request.category,
            sentiment=request.sentiment,
            min_rating=request.rating_min,
            max_rating=request.rating_max,
            product_name_contains=request.product or request.product_name_contains,
        )
        items = []
        for _, row in result.iterrows():
            item = {
                "id": str(row.get("doc_id")),
                "review_text": str(row.get("comment") or row.get("retrieval_text") or ""),
                "sentiment": str(row.get("predicted_sentiment", "unknown")),
                "rating": None if pd.isna(row.get("rating")) else row.get("rating"),
                "category": row.get("category"),
                "product_name": row.get("product_name"),
                "product_id": None if pd.isna(row.get("product_id")) else str(row.get("product_id")),
                "source": "faiss",
                "score_or_match_reason": f"score={float(row.get('score', 0.0)):.4f}",
                "score": float(row.get("score", 0.0)),
            }
            items.append(item)
        return items, warnings
    except Exception as exc:
        warnings.append(f"Dense retrieval unavailable; keyword fallback used. Reason: {exc}")
        return keyword_search(
            query,
            top_k=request.top_k,
            sentiment=request.sentiment,
            category=request.category,
            rating_min=request.rating_min,
            rating_max=request.rating_max,
            product=request.product or request.product_name_contains,
        ), warnings


def issue_hits_for_text(text: str) -> list[str]:
    return [name for name, spec in ISSUE_TAXONOMY.items() if contains_any(text, spec["keywords"])]


def summarize_themes(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for item in evidence:
        text = item.get("review_text", "")
        for issue in issue_hits_for_text(str(text)):
            counts[issue] = counts.get(issue, 0) + 1
            examples.setdefault(issue, [])
            if len(examples[issue]) < 2:
                examples[issue].append(str(text)[:220])
    ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return [
        {
            "issue": issue,
            "label_vi": ISSUE_TAXONOMY[issue]["label_vi"],
            "count": count,
            "example_reviews": examples.get(issue, []),
        }
        for issue, count in ranked
    ]


def estimate_confidence(query: str, evidence: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    terms = query_tokens(query)
    related = 0
    for item in evidence:
        text = normalize_text(item.get("review_text", ""))
        overlap = sum(1 for token in terms if token in text)
        issue_overlap = any(issue_hits_for_text(str(item.get("review_text", ""))))
        if overlap >= 1 or issue_overlap:
            related += 1
    if related >= 4:
        confidence = "high"
    elif related >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    reason = f"{related}/{len(evidence)} review truy xuất có dấu hiệu liên quan trực tiếp đến câu hỏi."
    return confidence, {
        "relevant_count_estimate": related,
        "total_evidence": len(evidence),
        "reason": reason,
    }


def suggested_actions_for(themes: list[dict[str, Any]]) -> list[str]:
    actions = []
    for theme in themes[:4]:
        action = ISSUE_TAXONOMY.get(theme["issue"], {}).get("action")
        if action and action not in actions:
            actions.append(action)
    if not actions:
        actions.append("Đọc thêm các review bằng chứng trước khi đưa ra quyết định vận hành.")
    return actions


def build_answer_text(
    summary: str,
    themes: list[dict[str, Any]],
    confidence: str,
    evidence_quality: dict[str, Any],
    actions: list[str],
    evidence: list[dict[str, Any]],
) -> str:
    theme_lines = [
        f"{idx}. {theme['label_vi']} ({theme['count']} review trong bằng chứng)"
        for idx, theme in enumerate(themes[:5], start=1)
    ] or ["1. Chưa phát hiện chủ đề nổi bật rõ ràng trong bằng chứng truy xuất."]
    action_lines = [f"- {action}" for action in actions]
    evidence_lines = [
        f"[{idx}] {item['review_text'][:260]}"
        for idx, item in enumerate(evidence[:5], start=1)
    ] or ["[1] Không có review bằng chứng phù hợp."]
    return "\n\n".join(
        [
            "Tóm tắt:\n" + summary,
            "Vấn đề/chủ đề chính:\n" + "\n".join(theme_lines),
            f"Mức độ tin cậy:\n{confidence.title()} — {evidence_quality['reason']}",
            "Gợi ý hành động:\n" + "\n".join(action_lines),
            "Bằng chứng:\n" + "\n".join(evidence_lines),
        ]
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    corpus_path = _first_existing_corpus()
    return {
        "status": "ok",
        "project": "ecommerce-sentiment-rag",
        "corpus_path": str(corpus_path.relative_to(PROJECT_DIR)),
        "total_reviews": int(len(load_reviews())),
    }


@app.get("/api/analytics/summary")
def analytics_summary() -> dict[str, Any]:
    df = load_reviews()
    sentiment_counts = df["_sentiment"].value_counts(dropna=False).to_dict()
    rating_counts = (
        df["_rating_numeric"].dropna().astype(int).astype(str).value_counts().sort_index().to_dict()
        if "_rating_numeric" in df
        else {}
    )
    avg_rating = df["_rating_numeric"].dropna().mean()
    categories = sorted(df["category"].dropna().astype(str).unique().tolist()) if "category" in df.columns else []
    warnings = []
    return {
        "total_reviews": int(len(df)),
        "sentiment_distribution": {str(k): int(v) for k, v in sentiment_counts.items()},
        "rating_distribution": {str(k): int(v) for k, v in rating_counts.items()},
        "average_rating": None if pd.isna(avg_rating) else float(round(avg_rating, 3)),
        "negative_review_count": int(sentiment_counts.get("negative", 0)),
        "available_filters": {
            "sentiments": sorted([str(value) for value in df["_sentiment"].dropna().unique()]),
            "ratings": sorted(rating_counts.keys()),
            "categories": categories,
            "has_product_names": bool("product_name" in df.columns and df["product_name"].notna().any()),
        },
        "warnings": warnings,
    }


@app.get("/api/reviews/search")
def review_search(
    q: str | None = None,
    sentiment: str | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
    category: str | None = None,
    product: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    df = apply_common_filters(load_reviews(), sentiment, category, rating_min, rating_max, product)
    match_reason = "metadata filters"
    if q:
        q_norm = normalize_text(q)
        text_match = df["_review_text"].fillna("").astype(str).map(normalize_text).str.contains(q_norm, regex=False)
        product_match = (
            df["product_name"].fillna("").astype(str).map(normalize_text).str.contains(q_norm, regex=False)
            if "product_name" in df.columns
            else pd.Series(False, index=df.index)
        )
        df = df[text_match | product_match]
        match_reason = f'keyword contains "{q}"'
    total = int(len(df))
    page = df.iloc[offset : offset + limit]
    return {
        "items": [review_item(row, match_reason=match_reason) for _, row in page.iterrows()],
        "total_estimate": total,
        "limit": limit,
        "offset": offset,
        "filters_used": {
            "q": q,
            "sentiment": sentiment,
            "rating_min": rating_min,
            "rating_max": rating_max,
            "category": category,
            "product": product,
        },
    }


@app.get("/api/analytics/issues")
def analytics_issues(
    sentiment: str | None = "negative",
    category: str | None = None,
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    df = apply_common_filters(load_reviews(), sentiment=sentiment, category=category)
    total = max(int(len(df)), 1)
    issues = []
    for issue, spec in ISSUE_TAXONOMY.items():
        mask = df["_review_text"].fillna("").astype(str).map(lambda text: contains_any(text, spec["keywords"]))
        hits = df[mask]
        examples = [
            review_item(row, match_reason=f"matched issue={issue}") for _, row in hits.head(2).iterrows()
        ]
        count = int(len(hits))
        if count:
            issues.append(
                {
                    "issue": issue,
                    "label_vi": spec["label_vi"],
                    "count": count,
                    "share": round(count / total, 4),
                    "example_reviews": examples,
                }
            )
    issues.sort(key=lambda item: item["count"], reverse=True)
    return {
        "issues": issues[:limit],
        "filters_used": {"sentiment": sentiment, "category": category, "limit": limit},
    }


@app.post("/api/rag")
def rag_answer(request: RagRequest) -> dict[str, Any]:
    query = (request.query or request.question or "").strip()
    if not query:
        return {
            "answer": "Vui lòng nhập câu hỏi phân tích review.",
            "summary": "",
            "main_themes": [],
            "suggested_actions": [],
            "confidence": "low",
            "evidence_quality": {"relevant_count_estimate": 0, "total_evidence": 0, "reason": "Không có câu hỏi."},
            "evidence": [],
            "warnings": ["Missing query/question."],
            "results": [],
        }

    evidence, warnings = dense_retrieve(request, query)
    themes = summarize_themes(evidence)
    confidence, evidence_quality = estimate_confidence(query, evidence)
    actions = suggested_actions_for(themes)
    if evidence_quality["relevant_count_estimate"] < 2:
        warnings.append("Bằng chứng truy xuất còn yếu; không nên xem câu trả lời là kết luận chắc chắn.")
    summary = (
        f"Tìm thấy {len(evidence)} review bằng chứng. "
        f"Chủ đề nổi bật nhất là {themes[0]['label_vi']}." if themes else
        f"Tìm thấy {len(evidence)} review nhưng chưa có chủ đề nổi bật rõ ràng."
    )
    answer = build_answer_text(summary, themes, confidence, evidence_quality, actions, evidence)
    return {
        "query": query,
        "answer": answer,
        "summary": summary,
        "main_themes": themes,
        "suggested_actions": actions,
        "confidence": confidence,
        "evidence_quality": evidence_quality,
        "evidence": evidence,
        "warnings": warnings,
        "results": evidence,
    }
