"""Review retrieval utilities for Module 4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


OUTPUT_COLUMNS = [
    "rank",
    "score",
    "doc_id",
    "category",
    "product_name",
    "product_id",
    "rating",
    "predicted_sentiment",
    "sentiment_confidence",
    "comment",
    "retrieval_text",
]


def resolve_project_dir(project_dir: str | None = None) -> Path:
    if project_dir:
        return Path(project_dir).expanduser().resolve()
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_path(project_dir: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_dir / path


class ReviewRetriever:
    """FAISS-backed retriever for Vietnamese e-commerce reviews."""

    def __init__(
        self,
        project_dir: str | Path | None = None,
        index_dir: str | Path = "models/module4",
        model_name: str | None = None,
    ) -> None:
        self.project_dir = resolve_project_dir(str(project_dir)) if project_dir else resolve_project_dir()
        self.index_dir = resolve_path(self.project_dir, index_dir)
        self.index_path = self.index_dir / "review_faiss.index"
        self.metadata_path = self.index_dir / "review_metadata.parquet"
        self.config_path = self.index_dir / "module4_index_config.json"

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata parquet not found: {self.metadata_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Index config not found: {self.config_path}")

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Missing Module 4 dependencies. Run: pip install -q -r requirements_module4.txt"
            ) from exc

        self.faiss = faiss
        self.config: dict[str, Any] = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_name = model_name or self.config.get("embedding_model", "intfloat/multilingual-e5-small")
        self.query_prefix = self.config.get("e5_query_prefix", "query: ")

        print(f"[retrieve] Loading FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.metadata_path).reset_index(drop=True)
        print(f"[retrieve] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        if self.index.ntotal != len(self.metadata):
            raise AssertionError(
                f"Smoke check failed: index.ntotal={self.index.ntotal} != metadata rows={len(self.metadata)}"
            )

    def _embed_query(self, query: str) -> np.ndarray:
        if not query or not str(query).strip():
            raise ValueError("query must be non-empty")
        query_text = self.query_prefix + str(query).strip()
        embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(embedding.astype("float32"))

    @staticmethod
    def _ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
        for column in OUTPUT_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        return df[OUTPUT_COLUMNS]

    @staticmethod
    def _filter(
        df: pd.DataFrame,
        category: str | None = None,
        sentiment: str | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        product_id: str | None = None,
        product_name_contains: str | None = None,
    ) -> pd.DataFrame:
        filtered = df
        if category and "category" in filtered.columns:
            filtered = filtered[
                filtered["category"].fillna("").astype(str).str.lower() == category.lower()
            ]
        if sentiment and "predicted_sentiment" in filtered.columns:
            filtered = filtered[
                filtered["predicted_sentiment"].fillna("").astype(str).str.lower()
                == sentiment.lower()
            ]
        if min_rating is not None and "rating" in filtered.columns:
            rating = pd.to_numeric(filtered["rating"], errors="coerce")
            filtered = filtered[rating >= min_rating]
        if max_rating is not None and "rating" in filtered.columns:
            rating = pd.to_numeric(filtered["rating"], errors="coerce")
            filtered = filtered[rating <= max_rating]
        if product_id and "product_id" in filtered.columns:
            filtered = filtered[
                filtered["product_id"].fillna("").astype(str).str.lower() == product_id.lower()
            ]
        if product_name_contains and "product_name" in filtered.columns:
            filtered = filtered[
                filtered["product_name"]
                .fillna("")
                .astype(str)
                .str.contains(product_name_contains, case=False, regex=False)
            ]
        return filtered

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        sentiment: str | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        product_id: str | None = None,
        product_name_contains: str | None = None,
    ) -> pd.DataFrame:
        top_k = max(int(top_k), 1)
        query_embedding = self._embed_query(query)
        candidate_k = min(max(top_k * 20, 100), int(self.index.ntotal))

        scores, indices = self.index.search(query_embedding, candidate_k)
        rows = self.metadata.iloc[indices[0]].copy().reset_index(drop=True)
        rows["score"] = scores[0]

        filtered = self._filter(
            rows,
            category=category,
            sentiment=sentiment,
            min_rating=min_rating,
            max_rating=max_rating,
            product_id=product_id,
            product_name_contains=product_name_contains,
        )

        if len(filtered) < top_k and candidate_k < self.index.ntotal:
            scores, indices = self.index.search(query_embedding, int(self.index.ntotal))
            rows = self.metadata.iloc[indices[0]].copy().reset_index(drop=True)
            rows["score"] = scores[0]
            filtered = self._filter(
                rows,
                category=category,
                sentiment=sentiment,
                min_rating=min_rating,
                max_rating=max_rating,
                product_id=product_id,
                product_name_contains=product_name_contains,
            )

        filtered = filtered.head(top_k).copy().reset_index(drop=True)
        filtered.insert(0, "rank", np.arange(1, len(filtered) + 1))
        return self._ensure_output_columns(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve reviews from Module 4 FAISS index.")
    parser.add_argument("--project_dir", default=None)
    parser.add_argument("--index_dir", default="models/module4")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category", default=None)
    parser.add_argument("--sentiment", default=None)
    parser.add_argument("--min_rating", type=float, default=None)
    parser.add_argument("--max_rating", type=float, default=None)
    parser.add_argument("--product_id", default=None)
    parser.add_argument("--product_name_contains", default=None)
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    results_dir = project_dir / "results" / "module4"
    results_dir.mkdir(parents=True, exist_ok=True)

    retriever = ReviewRetriever(project_dir=project_dir, index_dir=args.index_dir)

    general_smoke = retriever.retrieve("đánh giá sản phẩm", top_k=1)
    assert len(general_smoke) >= 1, "Smoke check failed: general retrieval returned no rows"

    result = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        category=args.category,
        sentiment=args.sentiment,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        product_id=args.product_id,
        product_name_contains=args.product_name_contains,
    )
    output_path = results_dir / "last_retrieval_results.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[retrieve] Query: {args.query}")
    print(f"[retrieve] Returned rows: {len(result)}")
    print(result.to_string(index=False))
    print(f"[retrieve] Saved CLI result: {output_path}")


if __name__ == "__main__":
    main()
