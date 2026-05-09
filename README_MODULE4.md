# README_MODULE4: Retrieval and RAG-style Review Insight

Module 4 builds the retrieval layer for Vietnamese e-commerce review analysis.

## Inputs
- data/processed/rag_corpus_module4.csv
- models/module4/review_faiss.index
- models/module4/review_metadata.parquet
- models/module4/module4_index_config.json

## Main functions
- Retrieve Vietnamese reviews using multilingual-E5 embeddings and FAISS.
- Support sentiment/category metadata filtering.
- Generate RAG-style evidence-grounded answers with citations.

## Outputs
- results/module4/rag_outputs.json
- results/module4/rag_outputs.md
- results/module4/retrieval_eval_summary.csv
- results/module4/manual_precision_at_k_sheet.csv

## Limitations
- Sentiment labels are inferred and may be noisy.
- Retrieval may return keyword-similar but irrelevant reviews.
- The answer generator is template-based, not a full LLM.
