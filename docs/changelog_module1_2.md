# Update Notes: Module 1 and Module 2 Source Inclusion

This update adds the official Module 1 and Module 2 source code to the final reproducible project package.

## Added Files
- `src/module1_data_layer.py`
- `src/module2_baseline_sentiment_classification.py`
- `scripts/run_module1.sh`
- `scripts/run_module2.sh`
- `README_MODULE1.md`
- `README_MODULE2.md`
- `requirements_module1_2.txt`

## Preservation Notes
- Existing processed data, Module 2 model artifacts, Module 4 FAISS index, Module 4 outputs, manual retrieval labels, and Module 5 reports were preserved.
- No model was retrained during this update.
- The FAISS index was not rebuilt.
- The final package rule excludes every `.zip` file case-insensitively to avoid nested archives.

## Full Pipeline
1. Module 1: Data Layer
2. Module 2: Baseline Sentiment Classification
3. Module 3: Sentiment Analytics
4. Module 4: Retrieval / RAG Pipeline
5. Module 5: Final Evaluation + Report Assembly
