# E-commerce Sentiment RAG

Vietnamese e-commerce review analytics project combining sentiment classification, retrieval, and evidence-centric review insight generation. The system cleans review data, trains classical sentiment baselines, prepares sentiment-enriched review corpora, builds a FAISS retrieval layer, and adds an aspect-aware hybrid retrieval and reranking pipeline.

This repository is designed as a course/project submission and a reproducible research-style prototype. Module 6 is evidence-centric: it retrieves and ranks review evidence, detects aspects, and summarizes aspect sentiment. It does not implement full LLM answer generation.

## Overview

The project targets Vietnamese product reviews from e-commerce settings. It supports:

- supervised sentiment classification over review text,
- sentiment analytics over RAG-style review corpora,
- dense FAISS retrieval over review evidence,
- hybrid BM25 + dense retrieval with Reciprocal Rank Fusion,
- optional CrossEncoder reranking with lexical fallback,
- rule-based aspect sentiment summaries for retrieved evidence.

## Main Features

- Data cleaning and schema normalization for sentiment and RAG corpora.
- Three-class sentiment mapping: `negative`, `neutral`, `positive`.
- Classical baselines: majority class, TF-IDF + Logistic Regression, TF-IDF + Linear SVM.
- RAG corpus enrichment with predicted sentiment labels.
- FAISS retrieval using multilingual embeddings.
- Module 6 hybrid retrieval: BM25, dense FAISS, RRF fusion, reranking, complaint-aware candidate expansion.
- Lightweight benchmark based on handcrafted keyword/aspect hit-rate queries.
- Final report assembly and manual retrieval evaluation support.

## Modules

- **Module 1: Data Layer**  
  Loads raw Excel/JSON data, cleans reviews, normalizes schema, creates sentiment splits, constructs the RAG corpus, and writes data quality reports.

- **Module 2: Baseline Sentiment Classification**  
  Trains majority, TF-IDF Logistic Regression, and TF-IDF Linear SVM baselines. Selects the best model using validation macro-F1 and evaluates once on the test split.

- **Module 3: Sentiment/RAG Data Preparation**  
  Applies the trained sentiment classifier to review corpora and creates sentiment analytics tables for categories, ratings, products, aspects, and review evidence.

- **Module 4: FAISS Retrieval**  
  Builds and uses dense review retrieval with multilingual E5 embeddings and FAISS. Includes lightweight RAG-style evidence answer demos without external LLM APIs.

- **Module 5: Final Evaluation + Reporting**  
  Aggregates Module 2-4 outputs, computes manual retrieval metrics when labels exist, generates figures, and assembles final report sections.

- **Module 6: Aspect-aware Hybrid Retrieval and Reranking**  
  Adds BM25 keyword retrieval, dense FAISS retrieval, RRF fusion, optional CrossEncoder reranking, complaint-aware candidate expansion, and rule-based aspect sentiment summaries. This is an evidence-centric pipeline, not full LLM generation.

## Project Structure

```text
ecommerce-sentiment-rag/
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

Create and activate a Python environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

For tests, install `pytest` if it is not already available:

```bash
python -m pip install pytest
```

Some Module 4/6 paths require existing local artifacts such as FAISS index files and processed corpus files. See `docs/artifact_policy.md` for recommended handling of large artifacts.

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

If the CrossEncoder model is unavailable, the reranker falls back to lexical scoring and logs a warning. The fallback is intentional so the demo remains runnable in offline environments.

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

## Data And Model Artifact Note

The repository may reference processed data, trained models, and FAISS indexes. These artifacts can be large and may have usage constraints depending on the original data source.

- Code and documentation are covered by the MIT license in this repository.
- Dataset/review corpus/model artifacts may have separate usage conditions.
- Large files such as FAISS indexes and processed corpora should be stored with Git LFS or GitHub Releases for public distribution.
- Final submission archives such as `ecommerce_sentiment_rag_final_submission.zip` should not be committed.

See [docs/artifact_policy.md](docs/artifact_policy.md) for details.

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

## Limitations

- Module 6 is evidence-centric and does not include full LLM answer generation.
- Rule-based aspect sentiment can miss sarcasm, typo-heavy text, and nuanced negation.
- The Module 6 benchmark uses handcrafted keyword/aspect hit-rate, not human relevance labels.
- Sentiment labels on RAG corpus are inferred from a baseline classifier and may be noisy.
- CrossEncoder reranking depends on model availability; lexical fallback is safer but less expressive.
- Large artifacts need careful publication through Git LFS or releases.

## Future Work

- Fine-tune PhoBERT/XLM-R or stronger Vietnamese encoders for sentiment classification.
- Train supervised aspect-based sentiment analysis.
- Train or fine-tune a query-review reranker using human relevance labels.
- Add controlled LLM answer generation with strict evidence citation and refusal behavior.
- Build a Streamlit or lightweight web demo.
- Expand manual retrieval evaluation with larger query sets and human labels.
