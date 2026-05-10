# Aspect-aware Hybrid Retrieval and Sentiment Analysis for E-commerce Reviews

Research/course project for Vietnamese e-commerce review analysis. The project combines data preprocessing, baseline sentiment classification, RAG corpus construction, dense FAISS retrieval, hybrid retrieval/reranking, aspect-level sentiment evidence, and a lightweight local backend/frontend demo layer.

This is not a production LLM chatbot and it does not implement full LLM answer generation. The RAG-style components retrieve and format evidence from reviews using deterministic, transparent logic.

## Overview

The repository supports:

- cleaning and normalizing Vietnamese e-commerce reviews,
- training classical sentiment baselines,
- enriching a review corpus with inferred sentiment metadata,
- retrieving review evidence with multilingual embeddings and FAISS,
- improving evidence retrieval with BM25, RRF fusion, and reranking,
- detecting simple aspect-level sentiment signals,
- inspecting results through a local API and dashboard.

The accepted current Module 4 indexed corpus contains **45,259 review documents**. The matching files are:

- `data/processed/rag_corpus_module4.csv`
- `models/module4/review_faiss.index`
- `models/module4/review_metadata.parquet`
- `models/module4/module4_index_config.json`

## Main Features

- Data cleaning and schema normalization for sentiment and retrieval corpora.
- Three-class sentiment mapping: `negative`, `neutral`, `positive`.
- Classical baselines: majority class, TF-IDF + Logistic Regression, TF-IDF + Linear SVM.
- RAG corpus enrichment with predicted sentiment labels.
- Dense retrieval using multilingual E5 embeddings and FAISS.
- Module 6 hybrid retrieval: BM25, dense retrieval, Reciprocal Rank Fusion, optional CrossEncoder reranking, complaint-aware candidate expansion, and lexical fallback.
- Rule-based aspect sentiment summaries for retrieved evidence.
- Lightweight dashboard for review exploration, issue discovery, suggested questions, and evidence inspection.

## Modules

- **Module 1: Data Layer**  
  Loads raw Excel/JSON data, cleans reviews, normalizes schema, creates sentiment splits, constructs the RAG corpus, and writes data quality reports.

- **Module 2: Baseline Sentiment Classification**  
  Trains majority, TF-IDF Logistic Regression, and TF-IDF Linear SVM baselines. Selects the best model using validation macro-F1 and evaluates once on the test split.

- **Module 3: Sentiment/RAG Data Preparation**  
  Applies the trained sentiment classifier to review corpora and creates descriptive sentiment analytics tables for categories, ratings, products, aspects, and review evidence.

- **Module 4: FAISS Retrieval**  
  Uses dense review retrieval with multilingual E5 embeddings and FAISS. It provides evidence retrieval and lightweight answer formatting without external LLM APIs.

- **Module 5: Final Evaluation + Reporting**  
  Aggregates Module 2-4 outputs, computes manual retrieval metrics when labels exist, generates figures, and assembles report sections.

- **Module 6: Aspect-aware Hybrid Retrieval and Reranking**  
  Adds BM25 retrieval, dense FAISS retrieval, RRF fusion, optional CrossEncoder reranking, complaint-aware expansion, and rule-based aspect sentiment evidence. This module is evidence-centric, not a full generative RAG system.

## Project Structure

```text
ecommerce-sentiment-rag/
  backend/                 # FastAPI demo API
  frontend/                # Static dashboard UI
  data/
    eval/
    processed/
    reports/
  docs/
    final_report/
  figures/
    module1/
    module2/
    module3/
    module4/
    module5/
    module6/
  models/
    module2/
    module4/
  reports/
  results/
    module2/
    module3/
    module4/
    module5/
    module6/
  scripts/
  src/
  tests/
  README_MODULE*.md
  requirements.txt
```

## Installation

```bash
python -m pip install -r requirements.txt
```

If you only want to run tests:

```bash
python -m pip install pytest
```

Large artifacts such as processed corpora and FAISS indexes may be better distributed through Git LFS or GitHub Releases. See `docs/artifact_policy.md`.

## Lightweight Product Demo Layer

The `backend/` and `frontend/` folders provide a simple local demo API/UI over the existing sentiment and retrieval artifacts. This layer is intended for demonstration, inspection, and course-project presentation. It is not a production deployment.

Backend endpoints:

- `GET /api/health`
- `GET /api/analytics/summary`
- `GET /api/analytics/issues`
- `GET /api/reviews/search`
- `POST /api/rag`

The frontend dashboard includes:

- overview cards for review count, sentiment ratio, average rating, and negative review count,
- simple sentiment distribution bars,
- top issue cards using keyword taxonomies,
- suggested business questions,
- review explorer with filters,
- evidence-oriented answer display with confidence and review citations.

Run the demo:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
frontend/index.html
```

## Health Checks

```bash
python scripts/check_project_health.py
python scripts/smoke_product_api.py
```

The health check verifies that the accepted Module 4 corpus and index configuration both report `45,259` documents.

## Run Tests

```bash
pytest -q
```

Expected current status:

```text
30 passed
```

Warnings from CrossEncoder fallback tests are expected and do not require internet access.

## Run Module 6 Demo

```bash
python scripts/run_module6_hybrid_rerank_demo.py \
  --query "Khách phàn nàn gì về đóng gói?" \
  --top-candidates 50 \
  --top-evidence 5 \
  --use-reranker true \
  --target-aspect dong_goi
```

If the CrossEncoder model is unavailable, the reranker falls back to lexical scoring and logs a warning. The fallback keeps the demo runnable offline.

## Run Module 6 Benchmark

```bash
python scripts/evaluate_module6_retrieval.py
```

Outputs:

```text
reports/module6_retrieval_eval.csv
figures/module6/module6_retrieval_comparison.png
```

The benchmark is a handcrafted keyword/aspect hit-rate evaluation. It is useful for diagnostics, but it is not a human relevance benchmark.

## Results Summary

Module 2 classical sentiment baseline:

- Logistic Regression test macro-F1: approximately `0.894`
- Linear SVM test macro-F1: approximately `0.910`

Module 6 retrieval benchmark, using handcrafted keyword/aspect hit-rate queries:

| Mode | keyword_hit@5 | keyword_hit@10 | aspect_hit@5 | aspect_hit@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.500 | 0.667 | 1.000 | 1.000 |
| Dense | 0.500 | 0.833 | 1.000 | 1.000 |
| Hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid + reranker | 1.000 | 1.000 | 1.000 | 1.000 |

These numbers are diagnostic and should not be presented as final benchmark performance.

## Data And Model Artifact Note

- Code and documentation are covered by the MIT license in this repository.
- Dataset, review corpus, and model artifacts may have separate usage conditions depending on the original data source.
- Final submission archives such as `ecommerce_sentiment_rag_final_submission.zip` should not be committed.

## Limitations

- The repository does not implement full LLM answer generation.
- Product demo answers use deterministic templates over retrieved evidence.
- Rule-based aspect sentiment can miss sarcasm, typo-heavy text, and nuanced negation.
- The Module 6 benchmark uses handcrafted keyword/aspect hit-rate, not human relevance labels.
- Sentiment labels on the retrieval corpus are inferred from a baseline classifier and may be noisy.
- CrossEncoder reranking depends on model availability; lexical fallback is safer but less expressive.

## Future Work

- Fine-tune PhoBERT/XLM-R or stronger Vietnamese encoders for sentiment classification.
- Train supervised aspect-based sentiment analysis.
- Train or fine-tune a query-review reranker using human relevance labels.
- Add controlled LLM generation in a future version with strict evidence citation and refusal behavior.
- Expand manual retrieval evaluation with larger query sets and human labels.
