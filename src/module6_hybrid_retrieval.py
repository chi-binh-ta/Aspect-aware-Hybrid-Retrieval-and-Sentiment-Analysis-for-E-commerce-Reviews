"""Module 6: Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

This module is intentionally independent from the existing Module 4 FAISS
pipeline. It can reuse a Module 4-style dense retriever when one is provided,
but it does not build or mutate any FAISS index.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:
    from module6_aspect_sentiment import enrich_result_with_aspects
    from module6_query_intent import build_expanded_queries
except ImportError:  # pragma: no cover - helps if this is later package-ized.
    from .module6_aspect_sentiment import enrich_result_with_aspects
    from .module6_query_intent import build_expanded_queries

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised when optional dependency is missing.
    BM25Okapi = None


class SimpleBM25Okapi:
    """Small offline BM25 fallback used when rank-bm25 is unavailable.

    The preferred path still uses ``rank_bm25.BM25Okapi`` when installed. This
    fallback keeps tests and demos runnable in offline grading environments. It
    implements the standard Okapi BM25 formula with document frequency, IDF,
    term frequency, k1=1.5, and b=0.75.
    """

    def __init__(
        self,
        corpus_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_count = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.doc_count, 1)
        self.term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}

        for doc in corpus_tokens:
            term_freq: dict[str, int] = {}
            for token in doc:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.term_freqs.append(term_freq)
            for token in term_freq:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.idf = {
            token: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for term_freq, doc_len in zip(self.term_freqs, self.doc_lengths, strict=True):
            score = 0.0
            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if freq <= 0:
                    continue
                denominator = freq + self.k1 * (
                    1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-12)
                )
                score += self.idf.get(token, 0.0) * (freq * (self.k1 + 1.0)) / denominator
            scores.append(float(score))
        return scores


ID_KEYS = ("review_id", "doc_id", "id", "comment_id")
TEXT_KEYS = ("text", "comment", "retrieval_text", "review", "retrieved_text")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def tokenize(text: object) -> list[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def get_first_text(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in record and record[key] is not None:
            text = str(record[key]).strip()
            if text:
                return text
    return ""


def get_review_id(record: dict[str, Any], fallback_index: int | None = None) -> str:
    value = get_first_text(record, ID_KEYS)
    if value:
        return value
    if fallback_index is None:
        return ""
    return f"review_{fallback_index}"


def get_review_text(record: dict[str, Any]) -> str:
    return get_first_text(record, TEXT_KEYS)


def canonicalize_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, review in enumerate(reviews):
        review_id = get_review_id(review, index)
        if review_id in seen:
            review_id = f"{review_id}__dup_{index}"
        seen.add(review_id)
        text = get_review_text(review)
        canonical.append(
            {
                "review_id": review_id,
                "text": text,
                "original": dict(review),
                "index": index,
                "tokens": tokenize(text),
            }
        )
    return canonical


def bm25_retrieve(
    query: str,
    reviews: list[dict[str, Any]],
    top_k: int = 50,
) -> list[dict[str, Any]]:
    return BM25ReviewRetriever(reviews).retrieve(query, top_k=top_k)


class BM25ReviewRetriever:
    """Reusable BM25 retriever for a fixed review corpus."""

    def __init__(self, reviews: list[dict[str, Any]]) -> None:
        self.canonical = canonicalize_reviews(reviews)
        corpus_tokens = [item["tokens"] for item in self.canonical]
        bm25_cls = BM25Okapi if BM25Okapi is not None else SimpleBM25Okapi
        self.bm25 = bm25_cls(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        return self._retrieve(query, top_k=top_k)

    def _retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        top_k = max(int(top_k), 0)
        if top_k == 0 or not self.canonical:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        normalized_query = normalize_text(query)

        ranked = []
        query_token_set = set(query_tokens)
        for index, (item, score) in enumerate(zip(self.canonical, scores, strict=True)):
            overlap_count = len(query_token_set & set(item["tokens"]))
            token_overlap = overlap_count > 0
            phrase_match = bool(normalized_query and normalized_query in normalize_text(item["text"]))
            if not token_overlap and not phrase_match:
                continue
            phrase_boost = 1e-6 if phrase_match else 0.0
            overlap_boost = 1e-4 * overlap_count
            ranked.append((float(score) + overlap_boost + phrase_boost, index, item))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, _index, item) in enumerate(ranked[:top_k], start=1):
            results.append(
                {
                    "review_id": item["review_id"],
                    "text": item["text"],
                    "bm25_rank": rank,
                    "bm25_score": score,
                    "source_flags": ["bm25"],
                    "original": item["original"],
                }
            )
        return results


def call_dense_retriever(
    dense_retriever: Any,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if dense_retriever is None or top_k <= 0:
        return []

    if hasattr(dense_retriever, "retrieve"):
        try:
            raw_results = dense_retriever.retrieve(query=query, top_k=top_k)
        except TypeError:
            raw_results = dense_retriever.retrieve(query, top_k)
    elif callable(dense_retriever):
        try:
            raw_results = dense_retriever(query=query, top_k=top_k)
        except TypeError:
            raw_results = dense_retriever(query, top_k)
    else:
        raise TypeError("dense_retriever must expose retrieve(...) or be callable.")

    return coerce_dense_results(raw_results)


def coerce_dense_results(raw_results: Any) -> list[dict[str, Any]]:
    if raw_results is None:
        return []

    if hasattr(raw_results, "to_dict"):
        rows = raw_results.to_dict(orient="records")
    elif isinstance(raw_results, dict) and "results" in raw_results:
        rows = list(raw_results["results"])
    else:
        rows = list(raw_results)

    dense_results: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        review_id = get_review_id(row, position - 1)
        text = get_review_text(row)
        rank_value = row.get("rank", position)
        try:
            dense_rank = int(rank_value)
        except (TypeError, ValueError):
            dense_rank = position
        dense_results.append(
            {
                "review_id": review_id,
                "text": text,
                "dense_rank": dense_rank,
                "dense_score": row.get("score"),
                "source_flags": ["dense"],
                "original": dict(row),
            }
        )
    dense_results.sort(key=lambda item: item["dense_rank"])
    return dense_results


class Module4DenseRetriever:
    """Thin wrapper around the existing Module 4 FAISS retriever.

    This wrapper only loads an existing index. It deliberately does not call
    ``module4_build_index.py`` or rebuild any artifacts.
    """

    def __init__(
        self,
        project_dir: str | Path | None = None,
        index_dir: str | Path = "models/module4",
        model_name: str | None = None,
    ) -> None:
        try:
            from module4_retrieve import ReviewRetriever
        except ImportError as exc:  # pragma: no cover - import path depends on runtime.
            raise ImportError(
                "Could not import Module 4 ReviewRetriever. Run from the project root "
                "or add src/ to PYTHONPATH."
            ) from exc

        self.retriever = ReviewRetriever(
            project_dir=project_dir,
            index_dir=index_dir,
            model_name=model_name,
        )

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        return self.retriever.retrieve(query=query, top_k=top_k).to_dict(orient="records")


def add_rrf_score(accumulator: dict[str, Any], source: str, rank: int, rrf_k: int) -> None:
    if rank <= 0:
        return
    accumulator["rrf_score"] += 1.0 / (float(rrf_k) + float(rank))
    if source not in accumulator["source_flags"]:
        accumulator["source_flags"].append(source)


def hybrid_retrieve(
    query: str,
    reviews: list[dict],
    dense_retriever=None,
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    final_top_k: int = 50,
    rrf_k: int = 60,
    bm25_retriever: BM25ReviewRetriever | None = None,
) -> list[dict]:
    """Retrieve reviews with BM25, optional dense retrieval, and RRF fusion."""

    if not str(query or "").strip():
        raise ValueError("query must be non-empty")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    final_top_k = max(int(final_top_k), 0)
    if final_top_k == 0:
        return []

    canonical_reviews = canonicalize_reviews(reviews)
    review_lookup = {item["review_id"]: item for item in canonical_reviews}

    bm25_retriever = bm25_retriever or BM25ReviewRetriever(reviews)
    expanded_queries = build_expanded_queries(query)
    bm25_result_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for query_index, expanded_query in enumerate(expanded_queries):
        source = "bm25" if query_index == 0 else "bm25_expanded"
        bm25_result_groups.append((source, bm25_retriever.retrieve(expanded_query, top_k=bm25_top_k)))
    dense_results = call_dense_retriever(dense_retriever, query, int(dense_top_k))

    fused: dict[str, dict[str, Any]] = {}

    def ensure_entry(result: dict[str, Any]) -> dict[str, Any]:
        review_id = str(result.get("review_id") or "")
        if not review_id:
            raise ValueError("retrieval result is missing review_id/doc_id/id")
        lookup_item = review_lookup.get(review_id)
        text = result.get("text") or (lookup_item or {}).get("text") or ""
        original = (lookup_item or {}).get("original") or result.get("original") or {}
        if review_id not in fused:
            fused[review_id] = {
                **dict(original),
                "review_id": review_id,
                "text": text,
                "bm25_rank": None,
                "dense_rank": None,
                "rrf_score": 0.0,
                "source_flags": [],
            }
        elif text and not fused[review_id].get("text"):
            fused[review_id]["text"] = text
        return fused[review_id]

    for source, bm25_results in bm25_result_groups:
        for result in bm25_results:
            entry = ensure_entry(result)
            rank = int(result["bm25_rank"])
            entry["bm25_rank"] = min(
                rank,
                entry["bm25_rank"] if isinstance(entry["bm25_rank"], int) else math.inf,
            )
            add_rrf_score(entry, source, rank, rrf_k)

    for result in dense_results:
        entry = ensure_entry(result)
        rank = int(result["dense_rank"])
        entry["dense_rank"] = min(
            rank,
            entry["dense_rank"] if isinstance(entry["dense_rank"], int) else math.inf,
        )
        add_rrf_score(entry, "dense", rank, rrf_k)

    def sort_key(item: dict[str, Any]) -> tuple[float, int, int, str]:
        bm25_rank = item["bm25_rank"] if isinstance(item["bm25_rank"], int) else 10**9
        dense_rank = item["dense_rank"] if isinstance(item["dense_rank"], int) else 10**9
        return (-float(item["rrf_score"]), min(bm25_rank, dense_rank), bm25_rank, item["review_id"])

    ranked = sorted(fused.values(), key=sort_key)[:final_top_k]
    for item in ranked:
        flags = []
        if "bm25" in item["source_flags"]:
            flags.append("bm25")
        if "bm25_expanded" in item["source_flags"]:
            flags.append("bm25_expanded")
        if "dense" in item["source_flags"]:
            flags.append("dense")
        item["source_flags"] = flags
    return [enrich_result_with_aspects(item) for item in ranked]


__all__ = [
    "Module4DenseRetriever",
    "BM25ReviewRetriever",
    "SimpleBM25Okapi",
    "bm25_retrieve",
    "build_expanded_queries",
    "hybrid_retrieve",
    "tokenize",
]
