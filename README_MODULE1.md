# README_MODULE1: Data Layer

Module 1 is the official data preparation layer for the Vietnamese e-commerce sentiment and RAG project.

## Inputs
- `data/raw/Data_sent.xlsx`
- `data/raw/RAG.zip`

The script can also locate these files in `/content` on Google Colab or in the current working directory, then copies them into `data/raw/`.

## Main Functions
- Load and clean sentiment reviews.
- Normalize schema from `Cmt` and `sentiment` to `text` and `rating`.
- Map ratings `1,2` to `negative`, `3` to `neutral`, and `4,5` to `positive`.
- Detect duplicate normalized reviews and remove conflicting duplicates from the main training data.
- Create stratified `train`, `valid`, and `test` splits.
- Extract JSON reviews from the RAG corpus and build retrieval documents without using the `Response` field in `retrieval_text`.
- Generate data quality reports and exploratory figures.

## Outputs
- `data/processed/sentiment_reviews.csv`
- `data/processed/sentiment_reviews_5class.csv`
- `data/processed/rag_documents.jsonl`
- `data/processed/rag_corpus.csv`
- `data/reports/data_summary.json`
- `data/reports/sentiment_label_distribution.csv`
- `data/reports/rag_category_split_distribution.csv`
- `figures/module1/`

## Run
```bash
bash scripts/run_module1.sh
```

Module 1 does not train any model.
