#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python src/module2_baseline_sentiment_classification.py
