# README_MODULE6: Aspect-aware Hybrid Retrieval and Reranking

Module 6 upgrades the retrieval layer into an evidence-centric pipeline for Vietnamese e-commerce review analysis. It compares keyword retrieval, dense retrieval, hybrid fusion, optional reranking, and rule-based aspect sentiment summaries. It is not a full LLM generation system and does not claim benchmark-level QA performance.

## Motivation

Dense-only retrieval can miss important short complaint phrases in e-commerce reviews, especially phrases such as `sai size`, `móp hộp`, `shop không trả lời`, or `giao sai`. These phrases are often sparse, colloquial, and highly diagnostic. Module 6 adds BM25 and RRF fusion so exact complaint language can complement semantic FAISS retrieval.

## Components

- **BM25 keyword retrieval**: uses `rank_bm25.BM25Okapi` over review text to catch exact complaint phrases.
- **Dense FAISS retrieval**: reuses the existing Module 4 FAISS index and metadata; it does not rebuild the index.
- **RRF fusion**: combines BM25 and dense ranks with Reciprocal Rank Fusion, so documents retrieved by both systems are promoted.
- **Cross-encoder reranker**: optionally reranks hybrid candidates. If the model is unavailable, the code falls back to a lexical reranker with a clear warning.
- **Aspect-based sentiment MVP**: rule-based aspect detection and sentiment classification for aspects such as `giao_hang`, `dong_goi`, `gia`, `dich_vu_shop`, `chat_luong`, and `mau_ma_size_mau`.

## Pipeline Diagram

```text
User query
   |
   +--> BM25 keyword retrieval
   |
   +--> Dense FAISS retrieval
             |
             v
      RRF rank fusion
             |
             v
   Optional CrossEncoder reranker
             |
             v
   Aspect detection + rule-based aspect sentiment
             |
             v
   Evidence list + aspect summary
```

## Run Demo

```bash
python scripts/run_module6_hybrid_rerank_demo.py \
  --query "Khách phàn nàn gì về giao hàng?" \
  --top-candidates 50 \
  --top-evidence 5 \
  --use-reranker true \
  --target-aspect giao_hang
```

Fast offline run without CrossEncoder loading:

```bash
python scripts/run_module6_hybrid_rerank_demo.py \
  --query "Có vấn đề gì về đóng gói không?" \
  --top-candidates 50 \
  --top-evidence 5 \
  --use-reranker false \
  --target-aspect dong_goi
```

Output:

```text
results/module6/last_hybrid_rerank_demo.json
```

## Sample Queries

- "Khách phàn nàn gì về giao hàng?"
- "Có vấn đề gì về đóng gói không?"
- "Review nào nói shop không trả lời?"
- "Khách có phàn nàn sai màu sai size không?"
- "Điểm tốt và xấu của sản phẩm là gì?"

## Run Benchmark

```bash
python scripts/evaluate_module6_retrieval.py
```

Inputs and outputs:

```text
data/eval/module6_queries.jsonl
reports/module6_retrieval_eval.csv
figures/module6/module6_retrieval_comparison.png
```

The benchmark is an MVP: it reports keyword/aspect hit-rates instead of full human relevance judgments.

## Current Benchmark Results

| Mode | keyword_hit@5 | keyword_hit@10 | aspect_hit@5 | aspect_hit@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.500 | 0.667 | 1.000 | 1.000 |
| Dense | 0.500 | 0.833 | 1.000 | 1.000 |
| Hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid + reranker | 1.000 | 1.000 | 1.000 | 1.000 |

In this small handcrafted keyword/aspect hit-rate benchmark, `hybrid` and `hybrid_reranker` perform best. The result should be read as a diagnostic comparison over a small query set, not as a human relevance benchmark.

## Limitations

- The benchmark uses keyword/aspect hit-rate, not manually judged relevance.
- Aspect sentiment is rule-based and can miss negation, sarcasm, or mixed opinions.
- Dense retrieval quality depends on the existing Module 4 embedding model and index.
- CrossEncoder reranking may fall back to lexical scoring if the model is not cached or cannot be loaded.
- The output is evidence-centric; it prepares grounded review evidence and summaries but is not a complete controlled LLM answer generation system.

## Future Work

- Train supervised aspect-based sentiment analysis (ABSA) on labeled review spans.
- Train or fine-tune a Vietnamese/e-commerce reranker with query-review relevance labels.
- Add stronger spam/noise filtering before retrieval and reranking.
- Add controlled LLM answer generation that cites retrieved evidence and refuses unsupported claims.
- Expand benchmark coverage with manually labeled relevance judgments.
