#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/module4_retrieve.py --query "khach hang phan nan gi ve giao hang?" --top_k 5 --sentiment negative
python src/module4_rag_pipeline.py --query "Khach hang phan nan gi nhieu nhat ve chat luong san pham?" --top_k 5 --sentiment negative
