#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/module4_prepare_inputs.py
python src/module4_build_index.py --batch_size 64
python src/module4_retrieve.py --query "khách hàng phàn nàn gì về giao hàng?" --top_k 5 --sentiment negative
python src/module4_rag_pipeline.py --query "Khách hàng phàn nàn gì nhiều nhất về chất lượng sản phẩm?" --top_k 5 --sentiment negative
python src/module4_evaluate.py

