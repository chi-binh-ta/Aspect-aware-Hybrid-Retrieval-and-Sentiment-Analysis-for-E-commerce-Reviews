# Module 6: Aspect-aware Hybrid Retrieval and Reranking

## Motivation

Module 4 provides a dense FAISS retrieval layer for review evidence. Dense retrieval is useful for semantic matching, but Vietnamese e-commerce reviews contain many short complaint markers and exact phrases that carry most of the user intent: `sai size`, `móp hộp`, `rách bao bì`, `shop không trả lời`, `giao sai`, or `giao chậm`. These phrases may be sparse, informal, and highly local. A dense-only retriever can retrieve semantically related but overly positive or generic reviews, which is not ideal when the user asks for concrete evidence about complaints.

Module 6 therefore adds a hybrid retrieval and reranking layer. The goal is diagnostic and practical: improve evidence retrieval quality before any downstream answer generation. This remains an evidence-centric pipeline, not a full LLM generation system.

## Why Dense-only Is Not Enough

Dense FAISS retrieval can retrieve reviews that are topically related to a query such as “đóng gói”, but the top reviews may say “đóng gói kĩ” or “không có gì phàn nàn” rather than “móp hộp” or “rách bao bì”. For review analytics, the exact complaint phrase often matters. Dense retrieval is also sensitive to the embedding model and may blur positive and negative variants of the same topic.

BM25 complements dense retrieval by preserving lexical precision. When a user asks for “sai size” or “shop không trả lời”, an exact keyword match is often more reliable than a semantic neighbor.

## BM25 Keyword Retrieval

`src/module6_hybrid_retrieval.py` implements BM25 retrieval with `rank_bm25.BM25Okapi`. Reviews are tokenized with a lightweight Unicode-aware tokenizer. BM25 is used to catch exact or near-exact complaint phrases:

- `sai size`
- `móp hộp`
- `shop không trả lời`
- `giao sai`
- `không rep`

BM25 results keep metadata such as `review_id`, `text`, `bm25_rank`, and `source_flags`.

## Dense FAISS Retrieval

Module 6 reuses the existing Module 4 dense retrieval stack through `Module4DenseRetriever`. It loads:

```text
models/module4/review_faiss.index
models/module4/review_metadata.parquet
models/module4/module4_index_config.json
```

It does not call `module4_build_index.py` and does not rebuild FAISS.

## RRF Fusion

Hybrid retrieval combines BM25 and dense results with Reciprocal Rank Fusion:

```text
RRF(doc) = sum(1 / (k + rank_i(doc)))
```

The default `k` is `60`. Documents that appear in both BM25 and dense lists receive contributions from both sources and are promoted. Each result includes:

- `review_id`
- `text`
- `bm25_rank`
- `dense_rank`
- `rrf_score`
- `source_flags`
- `detected_aspects`
- `aspect_sentiments`

## Cross-encoder Reranker

`src/module6_reranker.py` implements `CrossEncoderReranker`. It can load a `sentence-transformers` CrossEncoder, with the default model configurable through the constructor or `MODULE6_CROSS_ENCODER_MODEL`.

If the CrossEncoder cannot be loaded due to missing package, missing cache, or network issues, Module 6 falls back to lexical reranking and logs:

```text
CrossEncoder unavailable; falling back to lexical reranker.
```

The reranker computes:

```text
final_score = 0.70 * normalized_reranker_score
            + 0.20 * normalized_rrf_score
            + 0.10 * normalized_keyword_bonus
```

This reranking is still a lightweight diagnostic layer. It should not be interpreted as a trained production reranker unless a supervised relevance model is later fine-tuned.

## Aspect-based Sentiment MVP

`src/module6_aspect_sentiment.py` implements a rule-based aspect sentiment MVP. It detects aspects such as:

- `chat_luong`
- `giao_hang`
- `dong_goi`
- `gia`
- `dich_vu_shop`
- `mau_ma_size_mau`

It then assigns aspect sentiment with transparent positive and negative lexicons. If an aspect has both positive and negative cues, the result is `mixed`. If no clear signal exists, it is `neutral`.

This design is intentionally simple and inspectable. It is suitable for report-ready evidence grouping, but not a replacement for supervised ABSA.

## Pipeline Diagram

```text
User query
   |
   +--> BM25 keyword retrieval
   |
   +--> Dense FAISS retrieval
             |
             v
      Reciprocal Rank Fusion
             |
             v
   Optional CrossEncoder reranking
             |
             v
   Rule-based aspect detection
             |
             v
   Rule-based aspect sentiment
             |
             v
   Evidence-centric output:
     - top reviews
     - scores
     - source flags
     - aspects
     - aspect sentiment summary
```

## Demo

Run the end-to-end demo:

```bash
python scripts/run_module6_hybrid_rerank_demo.py \
  --query "Khách phàn nàn gì về giao hàng?" \
  --top-candidates 50 \
  --top-evidence 5 \
  --use-reranker true \
  --target-aspect giao_hang
```

Fast offline run:

```bash
python scripts/run_module6_hybrid_rerank_demo.py \
  --query "Có vấn đề gì về đóng gói không?" \
  --top-candidates 50 \
  --top-evidence 5 \
  --use-reranker false \
  --target-aspect dong_goi
```

The latest demo output is saved to:

```text
results/module6/last_hybrid_rerank_demo.json
```

## Benchmark

Run:

```bash
python scripts/evaluate_module6_retrieval.py
```

The benchmark reads:

```text
data/eval/module6_queries.jsonl
```

and writes:

```text
reports/module6_retrieval_eval.csv
figures/module6/module6_retrieval_comparison.png
```

The current MVP benchmark compares:

- `bm25`
- `dense`
- `hybrid`
- `hybrid_reranker`

Metrics:

- `keyword_hit@5`
- `keyword_hit@10`
- `aspect_hit@5`
- `aspect_hit@10`
- average hit-rate summaries by mode

## Current Benchmark Results

| Mode | keyword_hit@5 | keyword_hit@10 | aspect_hit@5 | aspect_hit@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.500 | 0.667 | 1.000 | 1.000 |
| Dense | 0.500 | 0.833 | 1.000 | 1.000 |
| Hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid + reranker | 1.000 | 1.000 | 1.000 | 1.000 |

In this small handcrafted keyword/aspect hit-rate evaluation, `hybrid` and `hybrid_reranker` perform best. The result indicates that complaint-aware expansion can improve keyword evidence coverage, but it is not a substitute for full manual relevance evaluation.

## Limitations

- The evaluation set is small and handcrafted.
- Keyword hit-rate is only a proxy for relevance.
- Aspect hit-rate can be high even when the retrieved review has the wrong polarity.
- Rule-based sentiment can miss negation, sarcasm, typos, and mixed opinions.
- CrossEncoder fallback is lexical, not neural.
- This module produces evidence and structured summaries; it does not implement full LLM generation.

## Future Work

- Train supervised ABSA for aspect detection and aspect sentiment classification.
- Train or fine-tune a Vietnamese/e-commerce reranker with human relevance labels.
- Add controlled LLM answer generation with explicit evidence citations and refusal behavior for weak evidence.
- Expand the benchmark with manually labeled query-review relevance.
- Improve noise filtering for spam, voucher/reward text, and unrelated review content.
