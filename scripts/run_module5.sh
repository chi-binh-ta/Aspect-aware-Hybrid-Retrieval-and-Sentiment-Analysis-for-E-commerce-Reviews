#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/module5_collect_results.py
python src/module5_manual_retrieval_eval.py
python src/module5_error_analysis.py
python src/module5_generate_figures.py
python src/module5_generate_report_sections.py

echo "Module 5 outputs:"
echo "  results/module5/final_metrics_summary.csv"
echo "  results/module5/final_project_summary.json"
echo "  results/module5/manual_precision_summary.csv"
echo "  results/module5/error_analysis.md"
echo "  figures/module5/"
echo "  docs/final_report/"
