"""Smoke checks for the product API without starting an external server."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    try:
        from fastapi.testclient import TestClient
        from backend.app import app
    except Exception as exc:
        print(f"[smoke_api] FAIL: backend import failed: {exc}")
        return 1

    client = TestClient(app)

    health = client.get("/api/health")
    require(health.status_code == 200, "/api/health failed")
    health_json = health.json()
    require("total_reviews" in health_json, "/api/health missing total_reviews")

    summary = client.get("/api/analytics/summary")
    require(summary.status_code == 200, "/api/analytics/summary failed")
    summary_json = summary.json()
    for key in [
        "total_reviews",
        "sentiment_distribution",
        "rating_distribution",
        "average_rating",
        "negative_review_count",
        "available_filters",
        "warnings",
    ]:
        require(key in summary_json, f"/api/analytics/summary missing {key}")

    issues = client.get("/api/analytics/issues?sentiment=negative&limit=5")
    require(issues.status_code == 200, "/api/analytics/issues failed")
    require("issues" in issues.json(), "/api/analytics/issues missing issues")

    reviews = client.get("/api/reviews/search?q=giao%20h%C3%A0ng&sentiment=negative&limit=5")
    require(reviews.status_code == 200, "/api/reviews/search failed")
    reviews_json = reviews.json()
    require("items" in reviews_json and "total_estimate" in reviews_json, "/api/reviews/search missing fields")

    rag = client.post(
        "/api/rag",
        json={"query": "Khach hang phan nan gi ve giao hang cham?", "top_k": 3, "sentiment": "negative", "use_dense": False},
    )
    require(rag.status_code == 200, "/api/rag failed")
    rag_json = rag.json()
    for key in ["answer", "summary", "main_themes", "suggested_actions", "confidence", "evidence_quality", "evidence"]:
        require(key in rag_json, f"/api/rag missing {key}")

    print("[smoke_api] PASS")
    print(f"[smoke_api] total_reviews={health_json['total_reviews']}")
    print(f"[smoke_api] issue_count={len(issues.json().get('issues', []))}")
    print(f"[smoke_api] review_search_items={len(reviews_json.get('items', []))}")
    print(f"[smoke_api] rag_confidence={rag_json.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
