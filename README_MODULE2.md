# README_MODULE2: Baseline Sentiment Classification

Module 2 is the official classical baseline sentiment classification layer.

## Input
- `data/processed/sentiment_reviews.csv`

## Main Functions
- Load the cleaned sentiment dataset from Module 1.
- Validate required columns and label/split values.
- Train a majority-class baseline.
- Train TF-IDF + Logistic Regression.
- Train TF-IDF + Linear SVM.
- Perform lightweight hyperparameter tuning on the validation split only.
- Select the best configuration by validation macro-F1.
- Evaluate the selected model on the test split only after model selection.
- Save predictions, classification reports, error-analysis CSVs, metrics, figures, and model artifacts.

## Outputs
- `models/module2/tfidf_logreg.joblib`
- `models/module2/tfidf_linearsvm.joblib`
- `models/module2/best_model.joblib`
- `models/module2/label_mapping.json`
- `results/module2/metrics_summary.csv`
- `results/module2/metrics_summary.json`
- `results/module2/tuning_results.csv`
- `results/module2/logistic_regression_classification_report.csv`
- `results/module2/linear_svm_classification_report.csv`
- `results/module2/logistic_regression_predictions.csv`
- `results/module2/linear_svm_predictions.csv`
- `results/module2/error_analysis_logistic_regression.csv`
- `results/module2/error_analysis_linear_svm.csv`
- `figures/module2/`

## Run
```bash
bash scripts/run_module2.sh
```

Module 2 does not use the RAG corpus and does not fine-tune transformer models.
