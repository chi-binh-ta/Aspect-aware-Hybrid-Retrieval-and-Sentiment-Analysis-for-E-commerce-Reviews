# Module 5: Final Evaluation, Error Analysis & Report Assembly

Module 5 summarizes the Vietnamese E-commerce Sentiment Classifier + RAG-based Review Insight Generator project. It reads existing outputs from Modules 2-4, creates report-ready metrics and figures, supports manual retrieval evaluation, and generates Vietnamese markdown sections for the final report.

## Required Inputs

Core inputs:

- `results/module2/metrics_summary.csv`
- `results/module2/linear_svm_classification_report.csv`
- `results/module2/logistic_regression_classification_report.csv`
- `results/module4/rag_outputs.json`
- `results/module4/retrieval_eval_summary.csv`
- `results/module4/manual_precision_at_k_sheet.csv`

Optional inputs:

- `results/module3/rag_sentiment_summary.json`
- `results/module3/overall_sentiment_distribution.csv`
- `results/module3/rating_sentiment_crosstab.csv`
- Other Module 3 analysis files, if available.

Missing optional files are recorded rather than treated as fatal errors.

## Manual Retrieval Relevance Labeling

Open:

```text
results/module4/manual_precision_at_k_sheet.csv
```

Fill the `is_relevant_manual` column:

- `1` means the retrieved review is relevant to the query.
- `0` means the retrieved review is not relevant.
- Leave `notes` for short comments if useful.

Do not change query IDs, ranks, or scores. After labeling, rerun Module 5 to compute Precision@k.

## Run Module 5

On Colab or Linux/macOS:

```bash
bash scripts/run_module5.sh
```

On Windows PowerShell, run the same Python commands manually:

```powershell
python src/module5_collect_results.py
python src/module5_manual_retrieval_eval.py
python src/module5_error_analysis.py
python src/module5_generate_figures.py
python src/module5_generate_report_sections.py
```

## Outputs

Main outputs:

- `results/module5/final_metrics_summary.csv`
- `results/module5/final_project_summary.json`
- `results/module5/manual_precision_summary.csv`
- `results/module5/manual_precision_overall.csv`, only when manual labels exist.
- `results/module5/error_analysis.md`
- `results/module5/final_discussion.md`
- `results/module5/final_conclusion.md`
- `figures/module5/*.png`
- `docs/final_report/report_skeleton.md`
- `docs/final_report/final_report_sections.md`

## Final Submission Package

Run:

```bash
python scripts/package_final_submission.py
```

This creates:

```text
ecommerce_sentiment_rag_final_submission.zip
artifact_manifest.json
```

The manifest contains the packaged file path, size in bytes, and SHA-256 hash. The package excludes caches, `.pyc` files, notebook checkpoints, and `data/raw/`.

## Using Outputs In The Final Report

- Use `final_metrics_summary.csv` for experiment tables.
- Use `figures/module5/` charts in the Experiments section.
- Use `error_analysis.md` for the Error Analysis section.
- Use `docs/final_report/final_report_sections.md`, `final_discussion.md`, and `final_conclusion.md` as polished Vietnamese report text.
