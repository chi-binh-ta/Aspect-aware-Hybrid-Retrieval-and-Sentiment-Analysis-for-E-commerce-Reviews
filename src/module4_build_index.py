"""Build the Module 4 FAISS retrieval index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for Module 4 retrieval.")
    parser.add_argument("--project_dir", default=None)
    parser.add_argument("--input", default="data/processed/rag_corpus_module4.csv")
    parser.add_argument("--index_dir", default="models/module4")
    parser.add_argument("--model_name", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    input_path = resolve_path(project_dir, args.input)
    index_dir = resolve_path(project_dir, args.index_dir)
    results_dir = project_dir / "results" / "module4"
    index_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "review_faiss.index"
    metadata_path = index_dir / "review_metadata.parquet"
    config_path = index_dir / "module4_index_config.json"
    summary_path = results_dir / "module4_build_summary.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Prepared Module 4 corpus not found: {input_path}")

    print(f"[build] Project dir: {project_dir}")
    print(f"[build] Loading corpus: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input corpus is empty.")
    if "retrieval_text" not in df.columns:
        raise KeyError("Input corpus must contain retrieval_text.")

    df["retrieval_text"] = df["retrieval_text"].fillna("").astype(str).str.strip()
    df = df[df["retrieval_text"].str.len() > 0].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No non-empty retrieval_text rows available for indexing.")

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
        print(f"[build] Using deterministic sample_size={args.sample_size}")

    passages = ("passage: " + df["retrieval_text"].astype(str)).tolist()

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing Module 4 dependencies. Run: pip install -q -r requirements_module4.txt"
        ) from exc

    print(f"[build] Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print(f"[build] Encoding {len(passages):,} documents with batch_size={args.batch_size}")
    embeddings = model.encode(
        passages,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))

    embedding_dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    df = df.reset_index(drop=True).copy()
    df.insert(0, "faiss_row_id", np.arange(len(df), dtype=int))

    faiss.write_index(index, str(index_path))
    df.to_parquet(metadata_path, index=False)

    config = {
        "embedding_model": args.model_name,
        "index_type": "IndexFlatIP",
        "similarity": "cosine_via_normalized_inner_product",
        "e5_passage_prefix": "passage: ",
        "e5_query_prefix": "query: ",
        "embedding_dim": embedding_dim,
        "num_documents": int(index.ntotal),
        "metadata_path": str(metadata_path),
        "index_path": str(index_path),
        "input_path": str(input_path),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_documents_indexed": int(index.ntotal),
        "metadata_rows": int(len(df)),
        "embedding_dim": embedding_dim,
        "model_name": args.model_name,
        "output_paths": {
            "index": str(index_path),
            "metadata": str(metadata_path),
            "config": str(config_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    assert index.ntotal == len(df), "Smoke check failed: index.ntotal != len(metadata)"

    print(f"[build] Number of documents indexed: {index.ntotal:,}")
    print(f"[build] Embedding dimension: {embedding_dim}")
    print(f"[build] Saved FAISS index: {index_path}")
    print(f"[build] Saved metadata: {metadata_path}")
    print(f"[build] Saved config: {config_path}")
    print(f"[build] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
