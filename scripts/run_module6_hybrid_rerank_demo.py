from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
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

from module6_aspect_sentiment import summarize_aspect_evidence
from module6_hybrid_retrieval import Module4DenseRetriever, hybrid_retrieve
from module6_query_intent import infer_query_intent
from module6_reranker import CrossEncoderReranker


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("--use-reranker must be true or false")


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
    print(f"[module6_demo] Loading review corpus: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Review corpus is empty: {input_path}")
    if "review_id" not in df.columns and "doc_id" in df.columns:
        df["review_id"] = df["doc_id"]
    if "text" not in df.columns:
        if "comment" in df.columns:
            df["text"] = df["comment"]
        elif "retrieval_text" in df.columns:
            df["text"] = df["retrieval_text"]
        else:
            raise KeyError("Corpus must contain one of text, comment, retrieval_text")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    print(f"[module6_demo] Loaded reviews: {len(df):,}")
    return df.to_dict(orient="records")


def build_dense_retriever(project_dir: Path):
    if os.environ.get("MODULE6_SKIP_DENSE", "").strip().lower() in {"1", "true", "yes"}:
        print("[module6_demo] MODULE6_SKIP_DENSE is set; using BM25-only hybrid fallback.")
        return None
    try:
        return Module4DenseRetriever(project_dir=project_dir)
    except Exception as exc:
        print(f"[module6_demo] Dense retriever unavailable; continuing with BM25 only. Reason: {exc}")
        return None


def normalize_target_aspect(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "giao_hang": "giao_hang",
        "giao_hàng": "giao_hang",
        "dong_goi": "dong_goi",
        "đóng_gói": "dong_goi",
        "gia": "gia",
        "giá": "gia",
        "chat_luong": "chat_luong",
        "chất_lượng": "chat_luong",
        "dich_vu_shop": "dich_vu_shop",
        "dịch_vụ_shop": "dich_vu_shop",
        "mau_ma_size_mau": "mau_ma_size_mau",
        "mẫu_mã_size_màu": "mau_ma_size_mau",
    }
    return aliases.get(normalized, normalized)


def filter_by_aspect(results: list[dict[str, Any]], target_aspect: str | None) -> list[dict[str, Any]]:
    if not target_aspect:
        return results
    filtered = [
        result
        for result in results
        if target_aspect in result.get("detected_aspects", [])
        or target_aspect in (result.get("aspect_sentiments") or {})
    ]
    return filtered or results


def score_for_display(result: dict[str, Any]) -> float:
    value = result.get("final_score", result.get("rrf_score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def short_text(text: object, limit: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def normalize_evidence_text(text: object) -> str:
    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_text: dict[str, dict[str, Any]] = {}
    first_position: dict[str, int] = {}
    for position, result in enumerate(results):
        key = normalize_evidence_text(result.get("text") or result.get("comment") or result.get("retrieval_text"))
        if not key:
            key = str(result.get("review_id") or result.get("doc_id") or position)
        if key not in first_position:
            first_position[key] = position
        current = best_by_text.get(key)
        if current is None or score_for_display(result) > score_for_display(current):
            best_by_text[key] = result
    deduped = list(best_by_text.items())
    deduped.sort(key=lambda item: (-score_for_display(item[1]), first_position[item[0]]))
    return [result for _key, result in deduped]


def aspect_line(result: dict[str, Any], target_aspect: str | None = None) -> str:
    aspect_sentiments = result.get("aspect_sentiments") or {}
    if target_aspect and target_aspect in aspect_sentiments:
        return f"{target_aspect}:{aspect_sentiments[target_aspect]}"
    if not aspect_sentiments:
        return "none:neutral"
    return ", ".join(f"{aspect}:{sentiment}" for aspect, sentiment in aspect_sentiments.items())


def sentiment_line(result: dict[str, Any], target_aspect: str | None = None) -> str:
    aspect_sentiments = result.get("aspect_sentiments") or {}
    if target_aspect and target_aspect in aspect_sentiments:
        return str(aspect_sentiments[target_aspect])
    if len(set(aspect_sentiments.values())) == 1 and aspect_sentiments:
        return next(iter(aspect_sentiments.values()))
    if aspect_sentiments:
        return "mixed"
    return str(result.get("predicted_sentiment") or "neutral")


def select_top_evidence(
    query: str,
    results: list[dict[str, Any]],
    target_aspect: str | None,
    top_evidence: int,
) -> list[dict[str, Any]]:
    top_evidence = max(int(top_evidence), 0)
    if top_evidence == 0:
        return []

    results = deduplicate_results(results)
    intent = infer_query_intent(query)
    if not (intent.get("is_complaint_query") and target_aspect):
        return results[:top_evidence]

    complaint_results: list[dict[str, Any]] = []
    other_results: list[dict[str, Any]] = []
    for result in results:
        aspect_sentiments = result.get("aspect_sentiments") or {}
        complaint_score = result.get("complaint_intent_score", 0.0)
        try:
            complaint_score = float(complaint_score)
        except (TypeError, ValueError):
            complaint_score = 0.0
        if complaint_score > 0 or aspect_sentiments.get(target_aspect) == "negative":
            complaint_results.append(result)
        else:
            other_results.append(result)

    if not complaint_results:
        return results[:top_evidence]
    top_results = complaint_results[:top_evidence]
    if len(top_results) < top_evidence:
        top_results.extend(other_results[: top_evidence - len(top_results)])
    return top_results


def print_results(query: str, results: list[dict[str, Any]], target_aspect: str | None) -> None:
    print()
    print(f"Query: {query}")
    print("Top evidence:")
    if not results:
        print("  No evidence found.")
    for index, result in enumerate(results, start=1):
        print(f"[{index}] score={score_for_display(result):.4f}")
        print(f"    aspect={aspect_line(result, target_aspect)}")
        print(f"    sentiment={sentiment_line(result, target_aspect)}")
        print(f"    source={','.join(result.get('source_flags', [])) or 'reranker'}")
        print(f"    review=\"{short_text(result.get('text'))}\"")

    summary = summarize_aspect_evidence(results)
    if target_aspect and target_aspect in summary:
        summary = {target_aspect: summary[target_aspect]}

    print()
    print("Aspect summary:")
    if not summary:
        print("- none: neutral")
        return
    for aspect, payload in summary.items():
        print(f"- {aspect}: {payload.get('sentiment', 'neutral')}")
        print("  evidence:")
        for item in payload.get("evidence", [])[:3]:
            print(f"    + {short_text(item.get('text'), limit=140)}")


def save_last_demo(project_dir: Path, query: str, results: list[dict[str, Any]], target_aspect: str | None) -> None:
    output_dir = project_dir / "results" / "module6"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "target_aspect": target_aspect,
        "results": results,
        "aspect_summary": summarize_aspect_evidence(results),
    }
    output_path = output_dir / "last_hybrid_rerank_demo.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[module6_demo] Saved demo output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 6 hybrid retrieval + optional rerank demo.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-candidates", type=int, default=50)
    parser.add_argument("--top-evidence", type=int, default=5)
    parser.add_argument("--use-reranker", type=parse_bool, default=True)
    parser.add_argument("--target-aspect", default=None)
    args = parser.parse_args()

    target_aspect = normalize_target_aspect(args.target_aspect)
    reviews = load_review_corpus(PROJECT_DIR)
    dense_retriever = build_dense_retriever(PROJECT_DIR)

    candidates = hybrid_retrieve(
        query=args.query,
        reviews=reviews,
        dense_retriever=dense_retriever,
        bm25_top_k=args.top_candidates,
        dense_top_k=args.top_candidates,
        final_top_k=args.top_candidates,
    )

    if args.use_reranker:
        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(args.query, candidates, top_k=args.top_candidates)
    else:
        ranked = candidates

    ranked = filter_by_aspect(ranked, target_aspect)
    top_results = select_top_evidence(args.query, ranked, target_aspect, args.top_evidence)
    print_results(args.query, top_results, target_aspect)
    save_last_demo(PROJECT_DIR, args.query, top_results, target_aspect)


if __name__ == "__main__":
    main()
