"""Lightweight evidence-grounded RAG-style review insight demo.

This module does not call any external LLM API. It retrieves review evidence,
computes simple aggregate signals, detects aspect keywords, and writes a
Vietnamese answer grounded in retrieved reviews.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from module4_retrieve import ReviewRetriever, resolve_project_dir

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ASPECT_KEYWORDS = {
    "chất lượng": ["chất lượng", "hỏng", "lỗi", "bền", "tốt", "xấu", "rách", "vỡ"],
    "giao hàng": ["giao hàng", "ship", "vận chuyển", "nhanh", "chậm", "lâu"],
    "đóng gói": ["đóng gói", "gói hàng", "hộp", "móp", "bọc"],
    "giá": ["giá", "rẻ", "đắt", "tiền", "đáng tiền"],
    "dịch vụ/shop": ["shop", "tư vấn", "phục vụ", "nhắn tin", "đổi trả"],
    "mẫu mã/size/màu": ["màu", "size", "mẫu", "kiểu", "ảnh"],
}

NOISE_PATTERNS = [
    "nhận xu",
    "lấy xu",
    "kiếm xu",
    "video chỉ mang tính chất",
    "hình ảnh chỉ mang tính chất",
    "trên thị trường hiện nay",
    "khách hàng thường dễ dàng",
    "anh đ.",
    "ngụ",
]

COMPLAINT_MARKERS = [
    "không",
    "lỗi",
    "hỏng",
    "tệ",
    "chậm",
    "thiếu",
    "sai",
    "móp",
    "vỡ",
    "rách",
    "bẩn",
    "thất vọng",
    "kém",
    "đổi trả",
    "hoàn hàng",
    "giao sai",
    "không trả lời",
]

NEGATIVE_QUERY_MARKERS = [
    "phàn nàn",
    "tiêu cực",
    "chê",
    "không hài lòng",
    "vấn đề",
    "điểm yếu",
    "lỗi",
    "hỏng",
]


def clean_excerpt(text: str, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_for_match(text: object) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def contains_any_marker(text: object, markers: list[str]) -> bool:
    normalized_text = normalize_for_match(text)
    return any(normalize_for_match(marker) in normalized_text for marker in markers)


def parse_numeric_rating(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def detect_sentiment_rating_conflict(sentiment: object, rating: object) -> bool:
    numeric_rating = parse_numeric_rating(rating)
    if numeric_rating is None:
        return False
    normalized_sentiment = normalize_for_match(sentiment)
    if normalized_sentiment == "negative" and numeric_rating >= 4:
        return True
    if normalized_sentiment == "positive" and numeric_rating <= 2:
        return True
    return False


def is_negative_rag_query(query: str, requested_sentiment: str | None = None) -> bool:
    if requested_sentiment and normalize_for_match(requested_sentiment) == "negative":
        return True
    return contains_any_marker(query, NEGATIVE_QUERY_MARKERS)


def add_evidence_quality_flags(evidence: pd.DataFrame) -> pd.DataFrame:
    evidence = evidence.copy()
    evidence["complaint_marker_detected"] = evidence["comment"].apply(
        lambda text: contains_any_marker(text, COMPLAINT_MARKERS)
    )
    evidence["sentiment_rating_conflict"] = evidence.apply(
        lambda row: detect_sentiment_rating_conflict(
            row.get("predicted_sentiment"), row.get("rating")
        ),
        axis=1,
    )
    evidence["rating_numeric"] = evidence["rating"].apply(parse_numeric_rating)
    evidence["strong_negative_evidence"] = evidence.apply(
        lambda row: normalize_for_match(row.get("predicted_sentiment")) == "negative"
        and (
            (row.get("rating_numeric") is not None and row.get("rating_numeric") <= 3)
            or bool(row.get("complaint_marker_detected"))
        ),
        axis=1,
    )
    return evidence


def build_evidence_items(ordered_evidence: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    evidence_items = []
    for idx, (_, row) in enumerate(ordered_evidence.head(limit).iterrows(), start=1):
        evidence_items.append(
            {
                "citation": f"[{idx}]",
                "doc_id": row.get("doc_id"),
                "score": float(row.get("score")) if pd.notna(row.get("score")) else None,
                "sentiment": row.get("predicted_sentiment"),
                "category": row.get("category"),
                "product_name": row.get("product_name"),
                "rating": row.get("rating"),
                "complaint_marker_detected": bool(row.get("complaint_marker_detected")),
                "sentiment_rating_conflict": bool(row.get("sentiment_rating_conflict")),
                "excerpt": clean_excerpt(row.get("comment", "")),
            }
        )
    return evidence_items


def detect_aspects(texts: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    combined = "\n".join(texts).lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        count = sum(combined.count(keyword.lower()) for keyword in keywords)
        if count > 0:
            counts[aspect] = int(count)
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def generate_answer(
    query: str,
    evidence: pd.DataFrame,
    requested_sentiment: str | None = None,
) -> dict[str, object]:
    non_empty = evidence[
        evidence["comment"].fillna("").astype(str).str.strip().str.len() > 0
    ].copy()
    noise_mask = non_empty["comment"].apply(lambda text: contains_any_marker(text, NOISE_PATTERNS))
    noise_filtered_count = int(noise_mask.sum())
    useful = non_empty.loc[~noise_mask].copy()
    useful = add_evidence_quality_flags(useful) if not useful.empty else useful

    negative_query = is_negative_rag_query(query, requested_sentiment)
    if negative_query and not useful.empty:
        strong = useful[useful["strong_negative_evidence"]].copy()
        weak = useful[~useful["strong_negative_evidence"]].copy()
        ordered_useful = pd.concat([strong, weak], ignore_index=True)
    else:
        ordered_useful = useful.reset_index(drop=True)

    strong_negative_count = int(useful["strong_negative_evidence"].sum()) if not useful.empty else 0
    weak_or_mixed_count = int(len(useful) - strong_negative_count)
    evidence_items = build_evidence_items(ordered_useful)

    if len(useful) < 2:
        answer = "Không đủ bằng chứng từ các review được truy hồi."
        return {
            "query": query,
            "answer": answer,
            "sentiment_distribution": {},
            "detected_aspects": {},
            "noise_filtered_count": noise_filtered_count,
            "strong_negative_count": strong_negative_count,
            "weak_or_mixed_count": weak_or_mixed_count,
            "evidence": evidence_items,
        }

    sentiment_counts = Counter(useful["predicted_sentiment"].fillna("unknown").astype(str))
    aspect_counts = detect_aspects(useful["comment"].fillna("").astype(str).tolist())

    top_sentiment = sentiment_counts.most_common(1)[0][0]
    sentiment_phrase = {
        "negative": "nghiêng về phản hồi tiêu cực",
        "neutral": "khá trung tính",
        "positive": "nghiêng về phản hồi tích cực",
    }.get(top_sentiment, "không rõ xu hướng")

    aspect_sentence = (
        ", ".join([f"{aspect} ({count})" for aspect, count in list(aspect_counts.items())[:4]])
        if aspect_counts
        else "chưa thấy aspect nổi bật theo bộ từ khóa đơn giản"
    )

    citations = " ".join(item["citation"] for item in evidence_items[:3])
    if negative_query and strong_negative_count < 2:
        answer_lines = [
            "Các review được truy hồi có liên quan đến chủ đề, nhưng chưa đủ bằng chứng tiêu cực rõ ràng để kết luận chắc chắn.",
            f"Các aspect xuất hiện nhiều trong bằng chứng gồm: {aspect_sentence}.",
            f"Bằng chứng tham khảo nằm ở các review {citations}.",
        ]
    elif negative_query:
        answer_lines = [
            f"Dựa trên {len(useful)} review được truy hồi, có {strong_negative_count} bằng chứng tiêu cực rõ ràng.",
            f"Các phàn nàn chính liên quan đến: {aspect_sentence}.",
            f"Bằng chứng chính nằm ở các review {citations}.",
            "Các phản hồi tiêu cực được ưu tiên trích dẫn khi sentiment là negative và rating thấp hoặc có dấu hiệu khiếu nại trong nội dung.",
        ]
    else:
        answer_lines = [
            f"Dựa trên {len(useful)} review được truy hồi, câu hỏi này {sentiment_phrase}.",
            f"Các aspect xuất hiện nhiều trong bằng chứng gồm: {aspect_sentence}.",
            f"Bằng chứng chính nằm ở các review {citations}.",
        ]
        if top_sentiment == "positive":
            answer_lines.append(
                "Các phản hồi tích cực tập trung vào trải nghiệm hài lòng hoặc đặc điểm sản phẩm được khen trong các trích dẫn."
            )
        elif top_sentiment == "negative":
            answer_lines.append(
                "Các phản hồi tiêu cực mô tả vấn đề trong nội dung review; nên kiểm tra các trích dẫn để xác nhận ngữ cảnh sản phẩm."
            )
        else:
            answer_lines.append(
                "Các review trung tính thường mô tả trải nghiệm ở mức chấp nhận được, ít nhấn mạnh khen hoặc chê mạnh."
            )

    answer_lines.append("Trích dẫn bằng chứng:")
    for item in evidence_items[:3]:
        answer_lines.append(
            f"{item['citation']} {item['excerpt']} "
            f"(sentiment={item['sentiment']}, category={item['category']}, rating={item['rating']})"
        )

    answer = "\n".join(answer_lines)
    if len(useful) >= 2:
        assert "[1]" in answer and "[2]" in answer, "Smoke check failed: answer lacks [1], [2] citations"

    return {
        "query": query,
        "answer": answer,
        "sentiment_distribution": dict(sentiment_counts),
        "detected_aspects": aspect_counts,
        "noise_filtered_count": noise_filtered_count,
        "strong_negative_count": strong_negative_count,
        "weak_or_mixed_count": weak_or_mixed_count,
        "evidence": evidence_items,
    }


def save_markdown(output: dict[str, object], path: Path) -> None:
    lines = [
        "# Module 4 RAG Demo Output",
        "",
        f"**Query:** {output['query']}",
        "",
        "## Answer",
        "",
        str(output["answer"]),
        "",
        "## Evidence",
        "",
    ]
    for item in output.get("evidence", []):
        lines.append(
            f"- {item['citation']} `{item['doc_id']}` | score={item['score']} | "
            f"sentiment={item['sentiment']} | {item['excerpt']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight RAG-style review insight demo.")
    parser.add_argument("--project_dir", default=None)
    parser.add_argument("--index_dir", default="models/module4")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category", default=None)
    parser.add_argument("--sentiment", default=None)
    parser.add_argument("--min_rating", type=float, default=None)
    parser.add_argument("--max_rating", type=float, default=None)
    parser.add_argument("--product_id", default=None)
    parser.add_argument("--product_name_contains", default=None)
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    results_dir = project_dir / "results" / "module4"
    results_dir.mkdir(parents=True, exist_ok=True)

    retriever = ReviewRetriever(project_dir=project_dir, index_dir=args.index_dir)
    evidence = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        category=args.category,
        sentiment=args.sentiment,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        product_id=args.product_id,
        product_name_contains=args.product_name_contains,
    )

    output = generate_answer(args.query, evidence, requested_sentiment=args.sentiment)
    output["retrieved_rows"] = int(len(evidence))
    output["filters"] = {
        "category": args.category,
        "sentiment": args.sentiment,
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
        "product_id": args.product_id,
        "product_name_contains": args.product_name_contains,
    }

    json_path = results_dir / "rag_outputs.json"
    md_path = results_dir / "rag_outputs.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    save_markdown(output, md_path)

    print("[rag] Answer:")
    print(output["answer"])
    print(f"[rag] Saved JSON: {json_path}")
    print(f"[rag] Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
