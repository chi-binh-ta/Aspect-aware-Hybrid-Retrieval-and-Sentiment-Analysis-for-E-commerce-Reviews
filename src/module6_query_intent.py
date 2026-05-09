"""Shared query-intent utilities for Module 6 retrieval and reranking."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


ASPECT_ALIASES: dict[str, list[str]] = {
    "dong_goi": ["đóng gói", "dong goi", "bao bì", "hộp", "bọc", "chống sốc"],
    "giao_hang": ["giao hàng", "ship", "vận chuyển"],
    "dich_vu_shop": ["shop", "tư vấn", "trả lời", "rep"],
    "mau_ma_size_mau": ["size", "màu", "mẫu", "khác hình"],
    "gia": ["giá", "rẻ", "đắt", "đáng tiền"],
    "chat_luong": ["chất lượng", "lỗi", "hỏng", "kém"],
}

COMPLAINT_QUERY_MARKERS = [
    "phàn nàn",
    "chê",
    "vấn đề",
    "lỗi",
    "bị",
    "có bị",
    "điểm yếu",
    "tiêu cực",
    "không hài lòng",
]

COMPLAINT_EXPANSION_TERMS: dict[str, list[str]] = {
    "dong_goi": [
        "móp hộp",
        "hộp móp",
        "móp",
        "méo",
        "bẹp",
        "rách bao bì",
        "bao bì rách",
        "rách",
        "đóng gói sơ sài",
        "sơ sài",
        "không chống sốc",
        "vỡ",
        "gãy",
    ],
    "giao_hang": [
        "giao chậm",
        "giao lâu",
        "ship chậm",
        "giao sai",
        "giao nhầm",
        "giao thiếu",
    ],
    "dich_vu_shop": [
        "shop không trả lời",
        "không trả lời",
        "không rep",
        "shop im",
        "hỗ trợ kém",
    ],
    "mau_ma_size_mau": [
        "sai màu",
        "sai size",
        "khác hình",
        "không giống ảnh",
        "nhầm màu",
        "nhầm size",
    ],
    "chat_luong": [
        "bị lỗi",
        "bị hỏng",
        "hàng lỗi",
        "hàng hỏng",
        "chất lượng kém",
        "kém chất lượng",
    ],
}

PACKAGING_COMPLAINT_MARKERS = COMPLAINT_EXPANSION_TERMS["dong_goi"]

COMPLAINT_MARKERS = sorted(
    {
        marker
        for markers in COMPLAINT_EXPANSION_TERMS.values()
        for marker in markers
    }
    | {
        "lỗi",
        "hỏng",
        "tệ",
        "chậm",
        "thiếu",
        "sai",
        "bẩn",
        "thất vọng",
        "kém",
        "đổi trả",
        "hoàn hàng",
    },
    key=len,
    reverse=True,
)

ASPECT_COMPLAINT_MARKERS = {
    **COMPLAINT_EXPANSION_TERMS,
    "gia": ["đắt", "không đáng tiền", "phí tiền", "giá cao"],
}

NEGATED_COMPLAINT_PHRASES = [
    "không có gì phàn nàn",
    "không có gì để phàn nàn",
    "không có gì đáng phàn nàn",
    "không có phàn nàn",
    "không phàn nàn",
    "k phàn nàn",
    "ko phàn nàn",
    "không thấy phàn nàn",
    "ko thấy phàn nàn",
    "ko có gì phàn nàn",
    "ko có gì đáng phàn nàn",
    "không có gì để chê",
    "không có cái gì để chê",
    "không có gì chê",
    "k có gì chê",
    "không có vấn đề gì",
    "không vấn đề gì",
    "chưa gặp vấn đề",
    "không có dấu hiệu bị",
    "không bị móp",
    "không rách",
    "không bị rách",
    "không bị",
]

NEGATION_CUES = ["không", "k", "ko", "kh", "khong", "chưa", "chua"]


def has_any_marker(text: object, markers: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(marker) in normalized for marker in markers)


def infer_target_aspects(query: str) -> list[str]:
    normalized = normalize_text(query)
    aspects = []
    for aspect, aliases in ASPECT_ALIASES.items():
        if any(normalize_text(alias) in normalized for alias in aliases):
            aspects.append(aspect)
    return aspects


def infer_query_intent(query: str) -> dict:
    is_complaint_query = has_any_marker(query, COMPLAINT_QUERY_MARKERS)
    return {
        "is_complaint_query": is_complaint_query,
        "target_aspects": infer_target_aspects(query),
        "requires_negative_evidence": is_complaint_query,
    }


def marker_is_intrinsically_negative(marker: str) -> bool:
    normalized = normalize_text(marker)
    return normalized.startswith(("không ", "ko ", "khong "))


def is_negated_occurrence(normalized_text: str, normalized_marker: str, start: int) -> bool:
    if marker_is_intrinsically_negative(normalized_marker):
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


def has_non_negated_marker(text: object, markers: list[str]) -> bool:
    normalized = normalize_text(text)
    for marker in sorted(markers, key=len, reverse=True):
        normalized_marker = normalize_text(marker)
        for match in re.finditer(re.escape(normalized_marker), normalized):
            if not is_negated_occurrence(normalized, normalized_marker, match.start()):
                return True
    return False


def has_negated_complaint_phrase(text: object) -> bool:
    normalized = normalize_text(text)
    if any(normalize_text(marker) in normalized for marker in NEGATED_COMPLAINT_PHRASES):
        return True
    loose_pairs = [
        ("không thấy", "phàn nàn"),
        ("ko thấy", "phàn nàn"),
        ("k thấy", "phàn nàn"),
        ("không có gì", "chê"),
        ("ko có gì", "chê"),
        ("k có gì", "chê"),
    ]
    for left, right in loose_pairs:
        left_index = normalized.find(normalize_text(left))
        right_index = normalized.find(normalize_text(right))
        if left_index >= 0 and right_index > left_index and right_index - left_index <= 96:
            return True
    return False


def has_target_aspect_complaint(text: str, target_aspects: list[str]) -> bool:
    if not target_aspects:
        return has_non_negated_marker(text, COMPLAINT_MARKERS)
    for aspect in target_aspects:
        markers = ASPECT_COMPLAINT_MARKERS.get(aspect, COMPLAINT_MARKERS)
        if has_non_negated_marker(text, markers):
            return True
    return False


def build_expanded_queries(query: str) -> list[str]:
    intent = infer_query_intent(query)
    expanded_queries = [str(query or "").strip()]
    if not intent["is_complaint_query"]:
        return [query for query in expanded_queries if query]

    for aspect in intent["target_aspects"]:
        terms = COMPLAINT_EXPANSION_TERMS.get(aspect, [])
        if not terms:
            continue
        aspect_text = " ".join(ASPECT_ALIASES.get(aspect, [aspect]))
        expansion = f"{aspect_text} {' '.join(terms)}".strip()
        if expansion and expansion not in expanded_queries:
            expanded_queries.append(expansion)
    return [query for query in expanded_queries if query]


__all__ = [
    "ASPECT_ALIASES",
    "ASPECT_COMPLAINT_MARKERS",
    "COMPLAINT_EXPANSION_TERMS",
    "COMPLAINT_MARKERS",
    "COMPLAINT_QUERY_MARKERS",
    "NEGATED_COMPLAINT_PHRASES",
    "PACKAGING_COMPLAINT_MARKERS",
    "build_expanded_queries",
    "has_any_marker",
    "has_negated_complaint_phrase",
    "has_non_negated_marker",
    "has_target_aspect_complaint",
    "infer_query_intent",
    "normalize_text",
]
