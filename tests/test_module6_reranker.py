from __future__ import annotations

import logging

from module6_reranker import CrossEncoderReranker, FALLBACK_WARNING


def fallback_reranker(caplog) -> CrossEncoderReranker:
    caplog.set_level(logging.WARNING)
    return CrossEncoderReranker(model_name="__force_fallback__")


def test_rerank_preserves_metadata(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {
            "review_id": "r1",
            "text": "Hộp bị móp khi nhận hàng.",
            "rrf_score": 0.21,
            "keyword_bonus": 0.7,
            "detected_aspects": ["đóng gói"],
            "product_id": "p1",
        }
    ]
    results = reranker.rerank("đóng gói bị móp", candidates, top_k=1)
    assert results[0]["review_id"] == "r1"
    assert results[0]["text"] == candidates[0]["text"]
    assert results[0]["rrf_score"] == candidates[0]["rrf_score"]
    assert results[0]["keyword_bonus"] == candidates[0]["keyword_bonus"]
    assert results[0]["detected_aspects"] == ["đóng gói"]
    assert results[0]["product_id"] == "p1"
    assert "reranker_score" in results[0]
    assert "final_score" in results[0]
    assert results[0]["rank_after_rerank"] == 1


def test_top_k_limits_output(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {"review_id": "r1", "text": "sản phẩm đẹp", "rrf_score": 0.1},
        {"review_id": "r2", "text": "hộp móp", "rrf_score": 0.2},
        {"review_id": "r3", "text": "rách bao bì", "rrf_score": 0.3},
    ]
    results = reranker.rerank("đóng gói", candidates, top_k=2)
    assert len(results) == 2
    assert [row["rank_after_rerank"] for row in results] == [1, 2]


def test_fallback_logs_warning_when_model_unavailable(caplog) -> None:
    reranker = fallback_reranker(caplog)
    assert reranker.using_fallback is True
    assert FALLBACK_WARNING in caplog.text


def test_packaging_complaint_query_prioritizes_negative_packaging_reviews(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {
            "review_id": "positive",
            "text": "Đóng gói đẹp, giao nhanh, sản phẩm dùng tốt.",
            "rrf_score": 0.9,
        },
        {
            "review_id": "mop_hop",
            "text": "Khách nhận hàng bị móp hộp, nhìn rất thất vọng.",
            "rrf_score": 0.1,
        },
        {
            "review_id": "rach_bao_bi",
            "text": "Bao bì bị rách bao bì, hộp méo khi giao tới.",
            "rrf_score": 0.2,
        },
    ]
    results = reranker.rerank("khách phàn nàn gì về đóng gói", candidates, top_k=3)
    assert results[0]["review_id"] in {"mop_hop", "rach_bao_bi"}
    assert results[0]["review_id"] != "positive"


def test_negative_packaging_query_penalizes_negated_complaint_review(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {
            "review_id": "actual_complaint",
            "text": "Hộp bị móp, bao bì rách.",
            "rrf_score": 0.1,
        },
        {
            "review_id": "negated_complaint",
            "text": "Đóng gói cẩn thận, không có gì phàn nàn.",
            "rrf_score": 0.9,
        },
    ]
    results = reranker.rerank("Khách phàn nàn gì về đóng gói?", candidates, top_k=2)
    assert results[0]["review_id"] == "actual_complaint"


def test_complaint_intent_prefers_mop_rach_over_k_phan_nan(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {
            "review_id": "actual_complaint",
            "text": "Hộp bị móp, bao bì rách.",
            "rrf_score": 0.1,
        },
        {
            "review_id": "no_complaint",
            "text": "Đóng gói sản phẩm tạm ổn, k phàn nàn nhiều.",
            "rrf_score": 0.9,
        },
    ]
    results = reranker.rerank("Khách phàn nàn gì về đóng gói?", candidates, top_k=2)
    assert results[0]["review_id"] == "actual_complaint"


def test_complaint_intent_prefers_so_sai_vo_over_positive_packaging(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {
            "review_id": "actual_complaint",
            "text": "Đóng gói sơ sài nên sản phẩm bị vỡ.",
            "rrf_score": 0.1,
        },
        {
            "review_id": "positive_packaging",
            "text": "Đóng gói chắc chắn, giao nhanh, rất hài lòng.",
            "rrf_score": 0.9,
        },
    ]
    results = reranker.rerank("Khách phàn nàn gì về đóng gói?", candidates, top_k=2)
    assert results[0]["review_id"] == "actual_complaint"


def test_complaint_intent_prefers_meo_bep_over_negated_packaging(caplog) -> None:
    reranker = fallback_reranker(caplog)
    candidates = [
        {
            "review_id": "actual_complaint",
            "text": "Hộp méo và bị bẹp khi nhận.",
            "rrf_score": 0.1,
        },
        {
            "review_id": "negated",
            "text": "Không bị móp, không rách bao bì.",
            "rrf_score": 0.9,
        },
    ]
    results = reranker.rerank("Khách phàn nàn gì về đóng gói?", candidates, top_k=2)
    assert results[0]["review_id"] == "actual_complaint"
