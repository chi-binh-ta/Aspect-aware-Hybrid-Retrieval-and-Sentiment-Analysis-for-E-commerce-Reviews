from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from module6_aspect_sentiment import enrich_result_with_aspects
from module6_hybrid_retrieval import (
    BM25ReviewRetriever,
    Module4DenseRetriever,
    call_dense_retriever,
    hybrid_retrieve,
    normalize_text,
)
from module6_reranker import CrossEncoderReranker


DEFAULT_EVAL_QUERIES = [
    {
        "query": "Khách phàn nàn gì về đóng gói?",
        "target_keywords": [
            "móp hộp",
            "rách bao bì",
            "hộp móp",
            "bao bì rách",
            "móp",
            "rách",
            "méo",
            "bẹp",
            "vỡ",
            "gãy",
            "sơ sài",
            "không chống sốc",
        ],
        "target_aspect": "dong_goi",
    },
    {
        "query": "Có review nào nói giao hàng chậm hoặc giao sai không?",
        "target_keywords": ["giao chậm", "giao lâu", "giao sai", "ship chậm"],
        "target_aspect": "giao_hang",
    },
    {
        "query": "Review nào nói shop không trả lời?",
        "target_keywords": ["shop không trả lời", "không trả lời", "không rep", "shop không rep"],
        "target_aspect": "dich_vu_shop",
    },
    {
        "query": "Khách có phàn nàn sai màu sai size không?",
        "target_keywords": ["sai size", "sai màu", "không đúng size", "không đúng màu"],
        "target_aspect": "mau_ma_size_mau",
    },
    {
        "query": "Khách nhận xét gì về giá và đáng tiền?",
        "target_keywords": ["giá rẻ", "đáng tiền", "giá ổn", "rẻ"],
        "target_aspect": "gia",
    },
    {
        "query": "Sản phẩm có bị lỗi hỏng hoặc chất lượng kém không?",
        "target_keywords": ["bị lỗi", "bị hỏng", "chất lượng kém", "hàng lỗi", "hàng hỏng"],
        "target_aspect": "chat_luong",
    },
]


def load_review_corpus(project_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        project_dir / "data" / "processed" / "rag_corpus_module4.csv",
        project_dir / "data" / "processed" / "rag_corpus.csv",
    ]
    input_path = next((path for path in candidates if path.exists()), None)
    if input_path is None:
        raise FileNotFoundError(
            "No review corpus found. Expected data/processed/rag_corpus_module4.csv "
            "or data/processed/rag_corpus.csv"
        )

    print(f"[module6_eval] Loading review corpus: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Review corpus is empty: {input_path}")

    if "review_id" not in df.columns and "doc_id" in df.columns:
        df["review_id"] = df["doc_id"]
    if "review_id" not in df.columns:
        df["review_id"] = [f"review_{index}" for index in range(len(df))]

    if "text" not in df.columns:
        if "comment" in df.columns:
            df["text"] = df["comment"]
        elif "retrieval_text" in df.columns:
            df["text"] = df["retrieval_text"]
        else:
            raise KeyError("Corpus must contain one of text, comment, retrieval_text")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    print(f"[module6_eval] Loaded reviews: {len(df):,}")
    return df.to_dict(orient="records")


def ensure_eval_file(eval_path: Path) -> None:
    if eval_path.exists():
        return
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as file:
        for row in DEFAULT_EVAL_QUERIES:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[module6_eval] Created handcrafted eval set: {eval_path}")


def load_eval_queries(eval_path: Path) -> list[dict[str, Any]]:
    ensure_eval_file(eval_path)
    queries: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("target_keywords"), list) or not row.get("target_aspect"):
                raise ValueError(
                    f"{eval_path}:{line_number} must contain target_keywords and target_aspect"
                )
            row.setdefault("query_id", f"q{len(queries) + 1:02d}")
            queries.append(row)
    if not queries:
        raise ValueError(f"No eval queries found in {eval_path}")
    print(f"[module6_eval] Loaded eval queries: {len(queries)}")
    return queries


def build_dense_retriever(project_dir: Path):
    if os.environ.get("MODULE6_SKIP_DENSE", "").strip().lower() in {"1", "true", "yes"}:
        print("[module6_eval] MODULE6_SKIP_DENSE is set; dense modes will be empty.")
        return None
    try:
        return Module4DenseRetriever(project_dir=project_dir)
    except Exception as exc:
        print(f"[module6_eval] Dense retriever unavailable. Dense modes will be empty. Reason: {exc}")
        return None


def normalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for result in results:
        row = dict(result)
        if "review_id" not in row:
            row["review_id"] = row.get("doc_id") or row.get("id")
        if "text" not in row or not str(row.get("text") or "").strip():
            row["text"] = row.get("comment") or row.get("retrieval_text") or row.get("review") or ""
        normalized.append(enrich_result_with_aspects(row))
    return normalized


def retrieve_by_mode(
    mode: str,
    query: str,
    reviews: list[dict[str, Any]],
    bm25_retriever: BM25ReviewRetriever,
    dense_retriever: Any,
    reranker: CrossEncoderReranker | None,
    top_candidates: int,
    eval_top_k: int,
) -> list[dict[str, Any]]:
    if mode == "bm25":
        return normalize_results(bm25_retriever.retrieve(query, top_k=eval_top_k))

    if mode == "dense":
        if dense_retriever is None:
            return []
        return normalize_results(call_dense_retriever(dense_retriever, query, top_k=eval_top_k))

    if mode in {"hybrid", "hybrid_reranker"}:
        candidates = hybrid_retrieve(
            query=query,
            reviews=reviews,
            dense_retriever=dense_retriever,
            bm25_top_k=top_candidates,
            dense_top_k=top_candidates,
            final_top_k=top_candidates,
            bm25_retriever=bm25_retriever,
        )
        if mode == "hybrid":
            return candidates[:eval_top_k]
        if reranker is None:
            reranker = CrossEncoderReranker(model_name="__force_fallback__")
        return reranker.rerank(query, candidates, top_k=eval_top_k)

    raise ValueError(f"Unknown mode: {mode}")


def contains_any_keyword(text: object, keywords: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return any(normalize_text(keyword) in normalized_text for keyword in keywords)


def keyword_hit(results: list[dict[str, Any]], keywords: list[str], top_k: int) -> int:
    top_results = results[:top_k]
    return int(any(contains_any_keyword(result.get("text", ""), keywords) for result in top_results))


def aspect_hit(results: list[dict[str, Any]], target_aspect: str, top_k: int) -> int:
    top_results = results[:top_k]
    for result in top_results:
        aspects = set(result.get("detected_aspects") or [])
        aspect_sentiments = result.get("aspect_sentiments") or {}
        if target_aspect in aspects or target_aspect in aspect_sentiments:
            return 1
    return 0


def compact_top_examples(results: list[dict[str, Any]], limit: int = 3) -> str:
    examples = []
    for result in results[:limit]:
        review_id = result.get("review_id") or result.get("doc_id") or "unknown"
        text = " ".join(str(result.get("text") or "").split())
        if len(text) > 90:
            text = text[:87].rstrip() + "..."
        examples.append(f"{review_id}: {text}")
    return " | ".join(examples)


def evaluate_modes(
    reviews: list[dict[str, Any]],
    bm25_retriever: BM25ReviewRetriever,
    queries: list[dict[str, Any]],
    dense_retriever: Any,
    reranker: CrossEncoderReranker | None,
    top_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = ["bm25", "dense", "hybrid", "hybrid_reranker"]
    per_query_rows: list[dict[str, Any]] = []
    max_eval_k = 10

    for mode in modes:
        print(f"[module6_eval] Evaluating mode: {mode}")
        for query_index, query_spec in enumerate(queries, start=1):
            query = str(query_spec["query"])
            target_keywords = [str(item) for item in query_spec.get("target_keywords", [])]
            target_aspect = str(query_spec["target_aspect"])
            results = retrieve_by_mode(
                mode=mode,
                query=query,
                reviews=reviews,
                bm25_retriever=bm25_retriever,
                dense_retriever=dense_retriever,
                reranker=reranker,
                top_candidates=top_candidates,
                eval_top_k=max_eval_k,
            )
            per_query_rows.append(
                {
                    "row_type": "query",
                    "mode": mode,
                    "query_id": query_spec.get("query_id", f"q{query_index:02d}"),
                    "query": query,
                    "target_aspect": target_aspect,
                    "target_keywords": json.dumps(target_keywords, ensure_ascii=False),
                    "retrieved_count": len(results),
                    "keyword_hit@5": keyword_hit(results, target_keywords, 5),
                    "keyword_hit@10": keyword_hit(results, target_keywords, 10),
                    "aspect_hit@5": aspect_hit(results, target_aspect, 5),
                    "aspect_hit@10": aspect_hit(results, target_aspect, 10),
                    "average_keyword_hit@5": "",
                    "average_keyword_hit@10": "",
                    "average_aspect_hit@5": "",
                    "average_aspect_hit@10": "",
                    "top_examples": compact_top_examples(results),
                    "notes": "" if results else "no results",
                }
            )

    per_query_df = pd.DataFrame(per_query_rows)
    summary_rows = []
    for mode, group in per_query_df.groupby("mode", sort=False):
        summary_rows.append(
            {
                "row_type": "average",
                "mode": mode,
                "query_id": "",
                "query": "",
                "target_aspect": "",
                "target_keywords": "",
                "retrieved_count": float(group["retrieved_count"].mean()),
                "keyword_hit@5": float(group["keyword_hit@5"].mean()),
                "keyword_hit@10": float(group["keyword_hit@10"].mean()),
                "aspect_hit@5": float(group["aspect_hit@5"].mean()),
                "aspect_hit@10": float(group["aspect_hit@10"].mean()),
                "average_keyword_hit@5": float(group["keyword_hit@5"].mean()),
                "average_keyword_hit@10": float(group["keyword_hit@10"].mean()),
                "average_aspect_hit@5": float(group["aspect_hit@5"].mean()),
                "average_aspect_hit@10": float(group["aspect_hit@10"].mean()),
                "top_examples": "",
                "notes": "",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return per_query_df, summary_df


def save_figure(summary_df: pd.DataFrame, figure_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[module6_eval] matplotlib unavailable; skipping figure. Reason: {exc}")
        return False

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.set_index("mode")[
        ["average_keyword_hit@5", "average_aspect_hit@5", "average_keyword_hit@10", "average_aspect_hit@10"]
    ]
    ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hit rate")
    ax.set_title("Module 6 Retrieval Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()
    print(f"[module6_eval] Saved figure: {figure_path}")
    return True


def print_console_report(summary_df: pd.DataFrame, per_query_df: pd.DataFrame) -> None:
    columns = [
        "mode",
        "average_keyword_hit@5",
        "average_keyword_hit@10",
        "average_aspect_hit@5",
        "average_aspect_hit@10",
    ]
    print()
    print("Module 6 retrieval benchmark")
    print(summary_df[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_row = summary_df.sort_values(
        ["average_keyword_hit@5", "average_aspect_hit@5", "average_keyword_hit@10"],
        ascending=False,
    ).iloc[0]
    print()
    print(f"Best mode by keyword_hit@5: {best_row['mode']}")

    success = per_query_df[
        (per_query_df["mode"] == best_row["mode"])
        & (per_query_df["keyword_hit@5"] == 1)
        & (per_query_df["aspect_hit@5"] == 1)
    ]
    failure = per_query_df[
        (per_query_df["mode"] == best_row["mode"])
        & ((per_query_df["keyword_hit@5"] == 0) | (per_query_df["aspect_hit@5"] == 0))
    ]
    if not success.empty:
        row = success.iloc[0]
        print(f"Success example: {row['query_id']} | {row['query']}")
        print(f"  Top examples: {row['top_examples']}")
    if not failure.empty:
        row = failure.iloc[0]
        print(f"Failure example: {row['query_id']} | {row['query']}")
        print(f"  Top examples: {row['top_examples']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Module 6 BM25, dense, hybrid, and hybrid+reranker retrieval."
    )
    parser.add_argument("--eval-file", default="data/eval/module6_queries.jsonl")
    parser.add_argument("--top-candidates", type=int, default=50)
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("MODULE6_CROSS_ENCODER_MODEL", "__force_fallback__"),
        help="CrossEncoder model name. Defaults to lexical fallback for reproducible offline benchmark.",
    )
    args = parser.parse_args()

    eval_path = PROJECT_DIR / args.eval_file
    reviews = load_review_corpus(PROJECT_DIR)
    queries = load_eval_queries(eval_path)
    print("[module6_eval] Building reusable BM25 index")
    bm25_retriever = BM25ReviewRetriever(reviews)
    dense_retriever = build_dense_retriever(PROJECT_DIR)
    reranker = CrossEncoderReranker(model_name=args.reranker_model)

    per_query_df, summary_df = evaluate_modes(
        reviews=reviews,
        bm25_retriever=bm25_retriever,
        queries=queries,
        dense_retriever=dense_retriever,
        reranker=reranker,
        top_candidates=max(int(args.top_candidates), 10),
    )

    output_path = PROJECT_DIR / "reports" / "module6_retrieval_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.concat([per_query_df, summary_df], ignore_index=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[module6_eval] Saved report: {output_path}")

    figure_path = PROJECT_DIR / "figures" / "module6" / "module6_retrieval_comparison.png"
    save_figure(summary_df, figure_path)
    print_console_report(summary_df, per_query_df)


if __name__ == "__main__":
    main()
