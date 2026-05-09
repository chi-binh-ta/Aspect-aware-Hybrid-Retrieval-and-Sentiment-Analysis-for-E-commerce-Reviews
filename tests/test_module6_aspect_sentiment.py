from __future__ import annotations

from module6_aspect_sentiment import (
    classify_aspect_sentiment,
    summarize_aspect_evidence,
)
from module6_hybrid_retrieval import hybrid_retrieve
from module6_reranker import CrossEncoderReranker


def test_hang_dep_nhung_giao_lau() -> None:
    result = classify_aspect_sentiment("Hàng đẹp nhưng giao lâu")
    assert result["chat_luong"] == "positive"
    assert result["giao_hang"] == "negative"


def test_hop_bi_mop_bao_bi_rach() -> None:
    result = classify_aspect_sentiment("Hộp bị móp, bao bì rách")
    assert result["dong_goi"] == "negative"


def test_negated_packaging_complaints_are_not_negative() -> None:
    result = classify_aspect_sentiment("Không bị móp, không rách bao bì")
    assert result["dong_goi"] != "negative"


def test_actual_packaging_complaints_are_negative() -> None:
    result = classify_aspect_sentiment("Hộp bị móp, bao bì rách")
    assert result["dong_goi"] == "negative"


def test_so_sai_vo_packaging_is_negative() -> None:
    result = classify_aspect_sentiment("Đóng gói sơ sài nên sản phẩm bị vỡ")
    assert result["dong_goi"] == "negative"


def test_positive_packaging_with_no_complaint_is_not_negative() -> None:
    result = classify_aspect_sentiment("Đóng gói chắc chắn, không có gì phàn nàn")
    assert result["dong_goi"] != "negative"


def test_meo_packaging_is_negative() -> None:
    result = classify_aspect_sentiment("Đóng gói về nhận thì nó méo hết")
    assert result["dong_goi"] == "negative"


def test_gay_packaging_is_negative() -> None:
    result = classify_aspect_sentiment("Shop đóng gói kỹ hơn tí, về gãy mất 1 cái chân")
    assert result["dong_goi"] == "negative"


def test_shop_tu_van_nhiet_tinh() -> None:
    result = classify_aspect_sentiment("Shop tư vấn nhiệt tình")
    assert result["dich_vu_shop"] == "positive"


def test_gia_re_nhung_chat_luong_kem() -> None:
    result = classify_aspect_sentiment("Giá rẻ nhưng chất lượng kém")
    assert result["gia"] == "positive"
    assert result["chat_luong"] == "negative"


def test_mixed_when_aspect_has_positive_and_negative_signal() -> None:
    result = classify_aspect_sentiment("Chất lượng tốt nhưng bị lỗi sau một ngày")
    assert result["chat_luong"] == "mixed"


def test_summarize_aspect_evidence_groups_by_aspect() -> None:
    summary = summarize_aspect_evidence(
        [
            {"review_id": "r1", "text": "Hộp bị móp, bao bì rách", "final_score": 0.9},
            {"review_id": "r2", "text": "Shop tư vấn nhiệt tình", "final_score": 0.8},
        ]
    )
    assert summary["dong_goi"]["sentiment"] == "negative"
    assert summary["dich_vu_shop"]["sentiment"] == "positive"
    assert summary["dong_goi"]["evidence"][0]["review_id"] == "r1"


def test_hybrid_results_include_aspect_sentiments() -> None:
    results = hybrid_retrieve(
        "móp hộp",
        [{"review_id": "r1", "text": "Hộp bị móp, bao bì rách"}],
        dense_retriever=None,
        final_top_k=1,
    )
    assert results[0]["detected_aspects"] == ["dong_goi"]
    assert results[0]["aspect_sentiments"]["dong_goi"] == "negative"


def test_reranker_results_include_aspect_sentiments(caplog) -> None:
    reranker = CrossEncoderReranker(model_name="__force_fallback__")
    results = reranker.rerank(
        "shop tư vấn",
        [{"review_id": "r1", "text": "Shop tư vấn nhiệt tình", "rrf_score": 0.1}],
        top_k=1,
    )
    assert results[0]["detected_aspects"] == ["dich_vu_shop"]
    assert results[0]["aspect_sentiments"]["dich_vu_shop"] == "positive"
