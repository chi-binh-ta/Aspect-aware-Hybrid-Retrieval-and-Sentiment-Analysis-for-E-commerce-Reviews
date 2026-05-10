#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[module4] Rebuilding Module 4 inputs and FAISS index."
echo "[module4] This is intentionally separate from demo scripts because it overwrites active artifacts."
python src/module4_prepare_inputs.py
python src/module4_build_index.py --batch_size 64
