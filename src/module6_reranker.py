"""Module 6 Cross-Encoder reranking with an offline lexical fallback.

The reranker is designed to consume results from ``module6_hybrid_retrieval``.
It does not call an LLM and does not rebuild any retrieval index.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import CrossEncoder
except Exception as exc:  # pragma: no cover - only exercised without dependency.
    CrossEncoder = None
    _CROSS_ENCODER_IMPORT_ERROR = exc
else:
    _CROSS_ENCODER_IMPORT_ERROR = None

try:
    from module6_hybrid_retrieval import normalize_text, tokenize
except ImportError:  # pragma: no cover - helps when imported as a package later.
    from .module6_hybrid_retrieval import normalize_text, tokenize

try:
    from module6_aspect_sentiment import classify_aspect_sentiment, detect_aspects
except ImportError:  # pragma: no cover - helps when imported as a package later.
    from .module6_aspect_sentiment import classify_aspect_sentiment, detect_aspects

try:
    import module6_query_intent as query_intent
except ImportError:  # pragma: no cover - helps when imported as a package later.
    from . import module6_query_intent as query_intent


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


LOGGER = logging.getLogger(__name__)
FALLBACK_WARNING = "CrossEncoder unavailable; falling back to lexical reranker."
DEFAULT_CROSS_ENCODER_MODEL = os.environ.get(
    "MODULE6_CROSS_ENCODER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

ASPECT_KEYWORDS = {
    "chất lượng": ["chất lượng", "hỏng", "lỗi", "bền", "rách", "vỡ", "xấu"],
    "giao hàng": ["giao hàng", "ship", "vận chuyển", "nhanh", "chậm", "lâu", "giao sai"],
    "đóng gói": ["đóng gói", "gói hàng", "hộp", "móp", "bao bì", "bọc", "rách bao bì"],
    "giá": ["giá", "rẻ", "đắt", "tiền", "đáng tiền"],
    "dịch vụ/shop": ["shop", "tư vấn", "phục vụ", "nhắn tin", "đổi trả", "không trả lời"],
    "mẫu mã/size/màu": ["màu", "size", "mẫu", "kiểu", "ảnh", "sai size"],
}

COMPLAINT_MARKERS = [
    "móp hộp",
    "hộp móp",
    "bị móp",
    "móp",
    "rách bao bì",
    "bao bì rách",
    "bị rách",
    "rách",
    "đóng gói sơ sài",
    "sơ sài",
    "không chống sốc",
    "vỡ",
    "gãy",
    "bẹp",
    "méo",
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
    "giao sai",
    "không trả lời",
    "không rep",
]

PACKAGING_COMPLAINT_MARKERS = [
    "móp hộp",
    "hộp móp",
    "bị móp",
    "móp",
    "rách bao bì",
    "bao bì rách",
    "bị rách",
    "rách",
    "đóng gói sơ sài",
    "sơ sài",
    "không chống sốc",
    "vỡ",
    "gãy",
    "bẹp",
    "méo",
]

ASPECT_COMPLAINT_MARKERS = {
    "dong_goi": PACKAGING_COMPLAINT_MARKERS,
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

NEGATIVE_QUERY_MARKERS = COMPLAINT_QUERY_MARKERS + ["hỏng"]

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
    "không vấn đề gì",
    "không có vấn đề gì",
    "không có dấu hiệu bị",
    "không bị",
    "không bị móp",
    "không rách",
    "không bị rách",
    "chưa gặp vấn đề",
]

NEGATION_CUES = ["không", "k", "ko", "kh", "khong", "chưa", "chua"]


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    finite_values = [value for value in values if value == value]
    if not finite_values:
        return [0.0 for _value in values]
    min_value = min(finite_values)
    max_value = max(finite_values)
    if max_value == min_value:
        fill_value = 1.0 if max_value > 0 else 0.0
        return [fill_value for _value in values]
    return [
        (value - min_value) / (max_value - min_value) if value == value else 0.0
        for value in values
    ]


def detected_aspects_for_text(text: object) -> list[str]:
    return detect_aspects(str(text or ""))


def has_any_marker(text: object, markers: list[str]) -> bool:
    return query_intent.has_any_marker(text, markers)


def marker_is_intrinsically_negative(marker: str) -> bool:
    return query_intent.marker_is_intrinsically_negative(marker)


def is_negated_occurrence(normalized_text: str, normalized_marker: str, start: int) -> bool:
    return query_intent.is_negated_occurrence(normalized_text, normalized_marker, start)


def has_non_negated_marker(text: object, markers: list[str]) -> bool:
    return query_intent.has_non_negated_marker(text, markers)


def has_negated_complaint_phrase(text: object) -> bool:
    return query_intent.has_negated_complaint_phrase(text)


def infer_query_intent(query: str) -> dict:
    return query_intent.infer_query_intent(query)


def has_target_aspect_complaint(text: str, target_aspects: list[str]) -> bool:
    return query_intent.has_target_aspect_complaint(text, target_aspects)


def evidence_complaint_score(query: str, text: str) -> float:
    intent = infer_query_intent(query)
    if not intent["is_complaint_query"]:
        return 0.0

    target_aspects = list(intent.get("target_aspects") or [])
    aspect_sentiments = classify_aspect_sentiment(text)
    has_real_complaint = has_target_aspect_complaint(text, target_aspects)

    score = 0.0
    if has_real_complaint:
        score += 1.5
    if has_negated_complaint_phrase(text):
        score -= 2.0
    if not has_real_complaint:
        for aspect in target_aspects:
            if aspect_sentiments.get(aspect) == "positive":
                score -= 0.5
                break
    return float(score)


def lexical_relevance_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)

    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)
    phrase_bonus = 1.0 if normalized_query and normalized_query in normalized_text else 0.0

    query_aspects = set(detected_aspects_for_text(query))
    text_aspects = set(detected_aspects_for_text(text))
    aspect_bonus = 0.6 * len(query_aspects & text_aspects)

    return float(overlap + phrase_bonus + aspect_bonus + evidence_complaint_score(query, text))


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def computed_keyword_bonus(query: str, text: str) -> float:
    query_aspects = set(detected_aspects_for_text(query))
    text_aspects = set(detected_aspects_for_text(text))
    if not query_aspects:
        return 0.0
    bonus = 0.5 * len(query_aspects & text_aspects)
    complaint_adjustment = evidence_complaint_score(query, text)
    if complaint_adjustment > 0:
        bonus += 0.5
    elif complaint_adjustment < 0:
        bonus -= 0.8
    return float(bonus)


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, device: str | None = None):
        self.model_name = model_name or DEFAULT_CROSS_ENCODER_MODEL
        self.device = device
        self.model: Any | None = None
        self.using_fallback = False

        if self.model_name in {"__fallback__", "__force_fallback__", "lexical"}:
            self._activate_fallback()
            return

        if CrossEncoder is None:
            self._activate_fallback(_CROSS_ENCODER_IMPORT_ERROR)
            return

        try:
            self.model = CrossEncoder(self.model_name, device=device)
        except Exception as exc:  # pragma: no cover - depends on cache/network.
            self._activate_fallback(exc)

    def _activate_fallback(self, reason: object | None = None) -> None:
        self.using_fallback = True
        if reason is not None:
            LOGGER.debug("CrossEncoder load failure: %r", reason)
        LOGGER.warning(FALLBACK_WARNING)
        warnings.warn(FALLBACK_WARNING, RuntimeWarning, stacklevel=2)

    def _predict_scores(self, query: str, candidates: list[dict[str, Any]], text_key: str) -> list[float]:
        if self.model is None or self.using_fallback:
            return [lexical_relevance_score(query, str(candidate.get(text_key, ""))) for candidate in candidates]

        pairs = [(query, str(candidate.get(text_key, ""))) for candidate in candidates]
        try:
            raw_scores = self.model.predict(pairs)
            return [float(score) for score in raw_scores]
        except Exception as exc:  # pragma: no cover - defensive runtime fallback.
            self._activate_fallback(exc)
            return [lexical_relevance_score(query, str(candidate.get(text_key, ""))) for candidate in candidates]

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int = 5,
    ) -> list[dict]:
        if not str(query or "").strip():
            raise ValueError("query must be non-empty")
        top_k = max(int(top_k), 0)
        if top_k == 0 or not candidates:
            return []

        candidate_copies = [dict(candidate) for candidate in candidates]
        reranker_scores = self._predict_scores(query, candidate_copies, text_key)
        rrf_scores = [coerce_float(candidate.get("rrf_score")) for candidate in candidate_copies]

        keyword_bonuses = []
        complaint_intent_scores = []
        for candidate in candidate_copies:
            text = str(candidate.get(text_key, ""))
            if "keyword_bonus" in candidate:
                keyword_bonus = coerce_float(candidate.get("keyword_bonus"))
            else:
                keyword_bonus = computed_keyword_bonus(query, text)
            candidate["keyword_bonus"] = keyword_bonus
            complaint_intent_score = evidence_complaint_score(query, text)
            candidate["complaint_intent_score"] = complaint_intent_score
            detected_aspects = candidate.get("detected_aspects")
            if "detected_aspects" not in candidate or detected_aspects is None or detected_aspects == "":
                candidate["detected_aspects"] = detected_aspects_for_text(text)
            if not isinstance(candidate.get("aspect_sentiments"), dict):
                candidate["aspect_sentiments"] = classify_aspect_sentiment(text)
            keyword_bonuses.append(keyword_bonus)
            complaint_intent_scores.append(complaint_intent_score)

        normalized_reranker = minmax_normalize(reranker_scores)
        normalized_rrf = minmax_normalize(rrf_scores)
        normalized_keyword = minmax_normalize(keyword_bonuses)

        enriched = []
        for index, candidate in enumerate(candidate_copies):
            final_score = (
                0.70 * normalized_reranker[index]
                + 0.20 * normalized_rrf[index]
                + 0.10 * normalized_keyword[index]
                + complaint_intent_scores[index]
            )
            result = dict(candidate)
            result.setdefault("review_id", candidate.get("doc_id") or candidate.get("id") or f"candidate_{index}")
            result.setdefault("text", str(candidate.get(text_key, "")))
            result.setdefault("rrf_score", rrf_scores[index])
            result["reranker_score"] = float(reranker_scores[index])
            result["final_score"] = float(final_score)
            enriched.append(result)

        enriched.sort(
            key=lambda item: (
                -float(item["final_score"]),
                -float(item["reranker_score"]),
                -coerce_float(item.get("rrf_score")),
                str(item.get("review_id")),
            )
        )
        output = enriched[:top_k]
        for rank, result in enumerate(output, start=1):
            result["rank_after_rerank"] = rank
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank Module 6 hybrid retrieval candidates.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidates_json", required=True, help="JSON file containing a list of candidate dicts.")
    parser.add_argument("--output_json", default="results/module6/reranked_results.json")
    parser.add_argument("--model_name", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    candidates_path = Path(args.candidates_json)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    reranker = CrossEncoderReranker(model_name=args.model_name, device=args.device)
    results = reranker.rerank(args.query, candidates, text_key=args.text_key, top_k=args.top_k)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[module6_reranker] Saved reranked results: {output_path}")


if __name__ == "__main__":
    main()
