"""Create retrieval evaluation examples for manual precision@k review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from module4_retrieve import ReviewRetriever, resolve_project_dir

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_QUERIES = [
    {"query_id": "q01", "query": "Khách hàng phàn nàn gì về chất lượng sản phẩm?", "sentiment": "negative"},
    {"query_id": "q02", "query": "Có những lỗi hỏng hoặc vấn đề độ bền nào được nhắc đến?", "sentiment": "negative"},
    {"query_id": "q03", "query": "Khách hàng gặp vấn đề gì về giao hàng hoặc vận chuyển?", "sentiment": "negative"},
    {"query_id": "q04", "query": "Review nào nói giao hàng nhanh hoặc chậm?", "sentiment": None},
    {"query_id": "q05", "query": "Khách hàng phàn nàn gì về đóng gói, hộp hoặc bao bì?", "sentiment": "negative"},
    {"query_id": "q06", "query": "Đóng gói sản phẩm có chắc chắn và đẹp không?", "sentiment": None},
    {"query_id": "q07", "query": "Khách hàng đánh giá giá cả và độ đáng tiền như thế nào?", "sentiment": None},
    {"query_id": "q08", "query": "Có review nào chê sản phẩm đắt hoặc không đáng tiền?", "sentiment": "negative"},
    {"query_id": "q09", "query": "Khách hàng nói gì về dịch vụ shop và tư vấn?", "sentiment": None},
    {"query_id": "q10", "query": "Các điểm mạnh của sản phẩm được khách hàng khen là gì?", "sentiment": "positive"},
    {"query_id": "q11", "query": "Các điểm yếu của sản phẩm được nhắc đến là gì?", "sentiment": "negative"},
    {"query_id": "q12", "query": "Review tiêu cực trong ngành hàng Electronic nói gì?", "category": "Electronic", "sentiment": "negative"},
    {"query_id": "q13", "query": "Review tích cực trong ngành hàng Fashion khen điều gì?", "category": "Fashion", "sentiment": "positive"},
    {"query_id": "q14", "query": "Review trung tính trong HealthBeauty mô tả trải nghiệm như thế nào?", "category": "HealthBeauty", "sentiment": "neutral"},
    {"query_id": "q15", "query": "Khách hàng nhận xét gì về mẫu mã, size hoặc màu sắc?", "sentiment": None},
]


def ensure_queries_file(path: Path) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_QUERIES, ensure_ascii=False, indent=2), encoding="utf-8")
        return DEFAULT_QUERIES
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval examples for manual evaluation.")
    parser.add_argument("--project_dir", default=None)
    parser.add_argument("--index_dir", default="models/module4")
    parser.add_argument("--queries_path", default="data/evaluation_queries.json")
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    queries_path = project_dir / args.queries_path
    results_dir = project_dir / "results" / "module4"
    results_dir.mkdir(parents=True, exist_ok=True)

    queries = ensure_queries_file(queries_path)
    retriever = ReviewRetriever(project_dir=project_dir, index_dir=args.index_dir)

    examples_path = results_dir / "retrieval_eval_examples.jsonl"
    summary_path = results_dir / "retrieval_eval_summary.csv"
    manual_sheet_path = results_dir / "manual_precision_at_k_sheet.csv"

    summary_rows = []
    manual_rows = []

    with examples_path.open("w", encoding="utf-8") as examples_file:
        for query_item in queries:
            query_id = query_item["query_id"]
            query = query_item["query"]
            filters = {
                "category": query_item.get("category"),
                "sentiment": query_item.get("sentiment"),
                "min_rating": query_item.get("min_rating"),
                "max_rating": query_item.get("max_rating"),
                "product_id": query_item.get("product_id"),
                "product_name_contains": query_item.get("product_name_contains"),
            }
            for k in [3, 5, 10]:
                result = retriever.retrieve(query=query, top_k=k, **filters)
                sentiment_counts = (
                    result["predicted_sentiment"].fillna("unknown").value_counts().to_dict()
                    if not result.empty
                    else {}
                )
                summary_rows.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "k": k,
                        "num_results": int(len(result)),
                        "avg_score": float(result["score"].mean()) if not result.empty else None,
                        "sentiment_counts": json.dumps(sentiment_counts, ensure_ascii=False),
                        "category_filter": filters["category"],
                        "sentiment_filter": filters["sentiment"],
                    }
                )

                record = {
                    "query_id": query_id,
                    "query": query,
                    "k": k,
                    "filters": filters,
                    "results": result.to_dict(orient="records"),
                }
                examples_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

                for _, row in result.iterrows():
                    manual_rows.append(
                        {
                            "query_id": query_id,
                            "query": query,
                            "k": k,
                            "rank": int(row["rank"]),
                            "score": float(row["score"]),
                            "doc_id": row.get("doc_id"),
                            "retrieved_text": row.get("comment") or row.get("retrieval_text"),
                            "is_relevant_manual": "",
                            "notes": "",
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    manual_df = pd.DataFrame(manual_rows)

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manual_df.to_csv(manual_sheet_path, index=False, encoding="utf-8-sig")

    print(f"[eval] Saved evaluation queries: {queries_path}")
    print(f"[eval] Saved examples: {examples_path}")
    print(f"[eval] Saved summary: {summary_path}")
    print(f"[eval] Saved manual precision@k sheet: {manual_sheet_path}")
    print(f"[eval] Queries: {len(queries)}")
    print(f"[eval] Manual rows: {len(manual_df)}")


if __name__ == "__main__":
    main()
