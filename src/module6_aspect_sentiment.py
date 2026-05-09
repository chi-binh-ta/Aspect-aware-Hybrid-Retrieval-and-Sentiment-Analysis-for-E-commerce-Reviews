"""Rule-based aspect sentiment MVP for Module 6.

This module intentionally uses transparent lexicons rather than an LLM. It is a
small diagnostic layer for grouping retrieved review evidence by aspect.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any


ASPECT_LEXICON: dict[str, list[str]] = {
    "chat_luong": [
        "chất lượng",
        "sản phẩm",
        "hàng",
        "đồ",
        "tốt",
        "đẹp",
        "ổn",
        "xấu",
        "lỗi",
        "hỏng",
        "bền",
        "kém",
    ],
    "giao_hang": [
        "giao hàng",
        "giao",
        "ship",
        "vận chuyển",
        "nhanh",
        "chậm",
        "lâu",
        "giao sai",
    ],
    "dong_goi": [
        "đóng gói",
        "gói hàng",
        "hộp",
        "móp",
        "bao bì",
        "bọc",
        "rách bao bì",
        "méo",
        "bẹp",
        "vỡ",
        "gãy",
        "chống sốc",
        "sơ sài",
    ],
    "gia": [
        "giá",
        "rẻ",
        "đắt",
        "tiền",
        "đáng tiền",
    ],
    "dich_vu_shop": [
        "shop",
        "tư vấn",
        "phục vụ",
        "nhắn tin",
        "đổi trả",
        "không trả lời",
        "không rep",
        "nhiệt tình",
    ],
    "mau_ma_size_mau": [
        "màu",
        "size",
        "mẫu",
        "kiểu",
        "ảnh",
        "sai size",
    ],
}

NEGATIVE_TERMS = [
    "tệ",
    "kém",
    "xấu",
    "lỗi",
    "hỏng",
    "chậm",
    "lâu",
    "móp",
    "rách",
    "sai",
    "thiếu",
    "không trả lời",
    "không rep",
    "thất vọng",
]

POSITIVE_TERMS = [
    "tốt",
    "đẹp",
    "ổn",
    "nhanh",
    "rẻ",
    "đáng tiền",
    "chắc chắn",
    "nhiệt tình",
    "hài lòng",
]

ASPECT_SENTIMENT_HINTS: dict[str, dict[str, list[str]]] = {
    "chat_luong": {
        "positive": ["tốt", "đẹp", "ổn", "chắc chắn", "hài lòng", "bền"],
        "negative": ["tệ", "kém", "xấu", "lỗi", "hỏng", "rách", "thiếu", "thất vọng"],
    },
    "giao_hang": {
        "positive": ["nhanh", "ổn", "hài lòng"],
        "negative": ["chậm", "lâu", "sai", "giao sai", "thất vọng"],
    },
    "dong_goi": {
        "positive": ["đẹp", "chắc chắn", "tốt", "ổn"],
        "negative": ["móp", "rách", "xấu", "thiếu", "thất vọng", "sơ sài", "vỡ", "gãy", "bẹp", "méo", "hỏng", "không chống sốc"],
    },
    "gia": {
        "positive": ["rẻ", "đáng tiền", "ổn", "hài lòng"],
        "negative": ["đắt", "thất vọng"],
    },
    "dich_vu_shop": {
        "positive": ["nhiệt tình", "tốt", "hài lòng", "nhanh"],
        "negative": ["không trả lời", "không rep", "chậm", "lâu", "tệ", "kém", "đổi trả"],
    },
    "mau_ma_size_mau": {
        "positive": ["đẹp", "ổn", "hài lòng"],
        "negative": ["sai", "xấu", "thiếu", "thất vọng"],
    },
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in text


def detect_aspects(text: str) -> list[str]:
    normalized = normalize_text(text)
    aspects = []
    for aspect, keywords in ASPECT_LEXICON.items():
        if any(contains_phrase(normalized, keyword) for keyword in keywords):
            aspects.append(aspect)
    return aspects


def has_any(text: str, terms: list[str]) -> bool:
    return any(contains_phrase(text, term) for term in terms)


NEGATION_CUES = [
    "không",
    "ko",
    "kh",
    "khong",
    "chưa",
    "chua",
]


def term_is_intrinsically_negative(term: str) -> bool:
    normalized = normalize_text(term)
    return normalized.startswith(("không ", "ko ", "khong "))


def is_negated_occurrence(normalized_text: str, normalized_term: str, start: int) -> bool:
    if term_is_intrinsically_negative(normalized_term):
        return False
    prefix = normalized_text[max(0, start - 48) : start]
    recent_tokens = prefix.split()[-8:]
    return any(
        cue in recent_tokens
        or prefix.endswith(f"{cue} ")
        or prefix.endswith(f"{cue} bị ")
        or prefix.endswith(f"{cue} bi ")
        for cue in NEGATION_CUES
    )


def contains_non_negated_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    for match in re.finditer(re.escape(normalized_phrase), normalized_text):
        if not is_negated_occurrence(normalized_text, normalized_phrase, match.start()):
            return True
    return False


def has_any_non_negated(text: str, terms: list[str]) -> bool:
    return any(contains_non_negated_phrase(text, term) for term in terms)


def aspect_sentiment_for_text(normalized_text: str, aspect: str) -> str:
    hints = ASPECT_SENTIMENT_HINTS.get(aspect, {})
    positive_terms = hints.get("positive") or POSITIVE_TERMS
    negative_terms = hints.get("negative") or NEGATIVE_TERMS
    has_positive = has_any(normalized_text, positive_terms)
    has_negative = has_any_non_negated(normalized_text, negative_terms)
    if has_positive and has_negative:
        return "mixed"
    if has_positive:
        return "positive"
    if has_negative:
        return "negative"
    return "neutral"


def classify_aspect_sentiment(text: str) -> dict:
    """Return aspect-level sentiment labels for a review text."""

    normalized = normalize_text(text)
    aspects = detect_aspects(normalized)
    return {aspect: aspect_sentiment_for_text(normalized, aspect) for aspect in aspects}


def summarize_sentiments(labels: list[str]) -> str:
    if not labels:
        return "neutral"
    counts = Counter(labels)
    non_neutral = {label: count for label, count in counts.items() if label != "neutral"}
    if not non_neutral:
        return "neutral"
    if "mixed" in non_neutral:
        return "mixed"
    if len(non_neutral) > 1:
        return "mixed"
    return max(non_neutral.items(), key=lambda item: item[1])[0]


def summarize_aspect_evidence(results: list[dict]) -> dict:
    """Group retrieved/reranked evidence snippets by aspect."""

    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"sentiments": [], "evidence": []})
    for result in results:
        text = str(result.get("text") or result.get("comment") or result.get("retrieval_text") or "")
        aspect_sentiments = result.get("aspect_sentiments")
        if not isinstance(aspect_sentiments, dict):
            aspect_sentiments = classify_aspect_sentiment(text)
        for aspect, sentiment in aspect_sentiments.items():
            grouped[aspect]["sentiments"].append(sentiment)
            grouped[aspect]["evidence"].append(
                {
                    "review_id": result.get("review_id") or result.get("doc_id"),
                    "text": text,
                    "sentiment": sentiment,
                    "rank": result.get("rank_after_rerank") or result.get("rank"),
                    "score": result.get("final_score") or result.get("rrf_score"),
                }
            )

    summary: dict[str, dict[str, Any]] = {}
    for aspect, payload in grouped.items():
        summary[aspect] = {
            "sentiment": summarize_sentiments(payload["sentiments"]),
            "evidence": payload["evidence"],
        }
    return dict(sorted(summary.items()))


def enrich_result_with_aspects(result: dict, text_key: str = "text") -> dict:
    enriched = dict(result)
    text = str(
        enriched.get(text_key)
        or enriched.get("comment")
        or enriched.get("retrieval_text")
        or ""
    )
    aspect_sentiments = classify_aspect_sentiment(text)
    enriched["detected_aspects"] = list(aspect_sentiments)
    enriched["aspect_sentiments"] = aspect_sentiments
    return enriched


__all__ = [
    "ASPECT_LEXICON",
    "NEGATIVE_TERMS",
    "POSITIVE_TERMS",
    "classify_aspect_sentiment",
    "detect_aspects",
    "enrich_result_with_aspects",
    "summarize_aspect_evidence",
]
