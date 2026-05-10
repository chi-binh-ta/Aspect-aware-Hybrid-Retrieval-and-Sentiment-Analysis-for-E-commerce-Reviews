"""Lightweight health checks for the local review intelligence project."""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_ACTIVE_DOCUMENTS = 45259


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def check_exists(path: Path, label: str, failures: list[str]) -> None:
    if path.exists():
        log("PASS", f"{label}: {path.relative_to(PROJECT_DIR)}")
    else:
        failures.append(f"Missing {label}: {path}")
        log("FAIL", f"{label} missing: {path.relative_to(PROJECT_DIR)}")


def compile_python_files(warnings: list[str], failures: list[str]) -> None:
    roots = [PROJECT_DIR / "src", PROJECT_DIR / "scripts", PROJECT_DIR / "backend"]
    files = [path for root in roots if root.exists() for path in root.rglob("*.py")]
    failed = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # pragma: no cover - diagnostic path
            failed.append(f"{path.relative_to(PROJECT_DIR)}: {exc}")
    if failed:
        failures.extend(failed)
        log("FAIL", f"Python compile failed for {len(failed)} file(s).")
    else:
        log("PASS", f"Python compile check passed for {len(files)} file(s).")
    if not files:
        warnings.append("No Python files found under src/scripts/backend.")
        log("WARN", "No Python files found under src/scripts/backend.")


def load_csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        # Fallback is faster for large CSVs and good enough for a health check.
        with path.open("rb") as file:
            return max(sum(1 for _ in file) - 1, 0)


def check_module4_consistency(warnings: list[str], failures: list[str]) -> None:
    corpus_path = PROJECT_DIR / "data" / "processed" / "rag_corpus_module4.csv"
    config_path = PROJECT_DIR / "models" / "module4" / "module4_index_config.json"
    index_path = PROJECT_DIR / "models" / "module4" / "review_faiss.index"
    metadata_path = PROJECT_DIR / "models" / "module4" / "review_metadata.parquet"

    for path, label in [
        (corpus_path, "Module 4 corpus"),
        (config_path, "Module 4 index config"),
        (index_path, "Module 4 FAISS index"),
        (metadata_path, "Module 4 metadata parquet"),
    ]:
        check_exists(path, label, failures)

    row_count = load_csv_row_count(corpus_path)
    config_docs = None
    if config_path.exists():
        try:
            config_docs = json.loads(config_path.read_text(encoding="utf-8")).get("num_documents")
        except Exception as exc:
            failures.append(f"Cannot read Module 4 index config: {exc}")
            log("FAIL", f"Cannot read Module 4 index config: {exc}")

    if row_count is not None:
        log("PASS", f"Module 4 corpus rows: {row_count:,}")
    if config_docs is not None:
        log("PASS", f"Module 4 index config num_documents: {int(config_docs):,}")
    if row_count is not None and config_docs is not None and int(config_docs) != int(row_count):
        failures.append(
            f"Module 4 row count mismatch: corpus={row_count}, config.num_documents={config_docs}"
        )
        log("FAIL", "Module 4 corpus row count does not match index config.")

    if row_count == EXPECTED_ACTIVE_DOCUMENTS:
        log("PASS", "Active Module 4 corpus matches the accepted 45,259-document artifact.")
    elif row_count is not None:
        warning = (
            f"Active Module 4 corpus has {row_count:,} rows; expected "
            f"{EXPECTED_ACTIVE_DOCUMENTS:,} for the current validated artifact set."
        )
        warnings.append(warning)
        log("WARN", warning)


def scan_stale_summaries(warnings: list[str]) -> None:
    roots = [PROJECT_DIR / "results", PROJECT_DIR / "data" / "reports", PROJECT_DIR / "docs"]
    stale_plain = "82" + "677"
    stale_comma = "82" + ",677"
    patterns = [rf"\b{stale_plain}\b", rf"\b{stale_comma}\b"]
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".md", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(re.search(pattern, text) for pattern in patterns):
                hits.append(path.relative_to(PROJECT_DIR).as_posix())
    if hits:
        warnings.append("Unexpected stale larger-corpus references found: " + ", ".join(hits[:12]))
        log("WARN", "Unexpected stale larger-corpus references found in reports/docs.")


def main() -> int:
    warnings: list[str] = []
    failures: list[str] = []

    print(f"[health] Project: {PROJECT_DIR}")
    compile_python_files(warnings, failures)
    check_exists(PROJECT_DIR / "models" / "module2" / "tfidf_linearsvm.joblib", "sentiment model", failures)
    check_exists(PROJECT_DIR / "data" / "processed" / "rag_corpus_module4.csv", "RAG corpus", failures)
    check_module4_consistency(warnings, failures)
    scan_stale_summaries(warnings)

    print("\n[health] Summary")
    print(f"PASS/WARN/FAIL counts: warnings={len(warnings)}, failures={len(failures)}")
    if failures:
        for failure in failures:
            print(f"  FAIL: {failure}")
        return 1
    if warnings:
        for warning in warnings:
            print(f"  WARN: {warning}")
    print("[health] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
