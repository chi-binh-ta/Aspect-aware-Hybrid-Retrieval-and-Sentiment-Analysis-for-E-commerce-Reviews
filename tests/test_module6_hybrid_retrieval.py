from __future__ import annotations

import module6_hybrid_retrieval as hybrid_module
from module6_hybrid_retrieval import hybrid_retrieve


def phrase_reviews() -> list[dict]:
    return [
        {"review_id": "r1", "text": "Áo giao sai size, mặc không vừa."},
        {"review_id": "r2", "text": "Hộp bị móp hộp rất nặng khi nhận hàng."},
        {"review_id": "r3", "text": "Shop không trả lời tin nhắn đổi trả."},
        {"review_id": "r4", "text": "Sản phẩm đẹp, giao nhanh, giá ổn."},
    ]


def test_bm25_catches_exact_phrase_sai_size() -> None:
    results = hybrid_retrieve("sai size", phrase_reviews(), dense_retriever=None, final_top_k=1)
    assert results[0]["review_id"] == "r1"
    assert results[0]["source_flags"] == ["bm25"]
    assert results[0]["bm25_rank"] == 1


def test_bm25_catches_exact_phrase_mop_hop() -> None:
    results = hybrid_retrieve("móp hộp", phrase_reviews(), dense_retriever=None, final_top_k=1)
    assert results[0]["review_id"] == "r2"
    assert results[0]["source_flags"] == ["bm25"]


def test_bm25_catches_exact_phrase_shop_khong_tra_loi() -> None:
    results = hybrid_retrieve(
        "shop không trả lời",
        phrase_reviews(),
        dense_retriever=None,
        final_top_k=1,
    )
    assert results[0]["review_id"] == "r3"
    assert results[0]["source_flags"] == ["bm25"]


class FakeDenseRetriever:
    def retrieve(self, query: str, top_k: int = 50) -> list[dict]:
        return [
            {"review_id": "r2", "text": "alpha beta second", "rank": 1, "score": 0.91},
            {"review_id": "r3", "text": "dense only", "rank": 2, "score": 0.70},
        ][:top_k]


def test_rrf_prioritizes_document_seen_by_bm25_and_dense() -> None:
    reviews = [
        {"review_id": "r1", "text": "alpha beta first"},
        {"review_id": "r2", "text": "alpha beta second"},
        {"review_id": "r3", "text": "dense only"},
    ]
    results = hybrid_retrieve(
        "alpha beta",
        reviews,
        dense_retriever=FakeDenseRetriever(),
        bm25_top_k=2,
        dense_top_k=2,
        final_top_k=3,
    )
    assert results[0]["review_id"] == "r2"
    assert results[0]["source_flags"] == ["bm25", "dense"]
    assert results[0]["bm25_rank"] is not None
    assert results[0]["dense_rank"] == 1


def test_hybrid_retrieve_does_not_crash_without_dense_retriever() -> None:
    results = hybrid_retrieve("giao nhanh", phrase_reviews(), dense_retriever=None, final_top_k=3)
    assert results
    assert all("dense" not in row["source_flags"] for row in results)
    assert all("rrf_score" in row for row in results)


def test_bm25_fallback_works_when_rank_bm25_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(hybrid_module, "BM25Okapi", None)
    results = hybrid_module.bm25_retrieve("móp hộp", phrase_reviews(), top_k=1)
    assert results[0]["review_id"] == "r2"


def test_build_expanded_queries_for_packaging_complaints() -> None:
    expanded = hybrid_module.build_expanded_queries("Khách phàn nàn gì về đóng gói?")
    assert any("móp" in query and "rách" in query and "vỡ" in query for query in expanded)


def test_bm25_expansion_adds_packaging_complaint_candidate() -> None:
    reviews = [
        {"review_id": "positive", "text": "Đóng gói cẩn thận, không có gì phàn nàn."},
        {"review_id": "complaint", "text": "Hộp bị móp, bao bì rách."},
    ]
    results = hybrid_retrieve(
        "Khách phàn nàn gì về đóng gói?",
        reviews,
        dense_retriever=None,
        bm25_top_k=1,
        final_top_k=2,
    )
    complaint = next(row for row in results if row["review_id"] == "complaint")
    assert "bm25_expanded" in complaint["source_flags"]
