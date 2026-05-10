# Vietnamese E-commerce Review Intelligence Assistant Demo

This demo turns the technical sentiment/RAG pipeline into a product-style dashboard for sellers, customer-service teams, and business analysts.

## Product Positioning

The application helps users inspect Vietnamese e-commerce reviews, find recurring complaints, search the original evidence, and ask evidence-grounded business questions. It is a local university-project prototype and does not call paid LLM APIs.

## Run

```bash
python scripts/check_project_health.py
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
frontend/index.html
```

## Demo Scenario

1. Open the dashboard.
2. Inspect summary cards: total reviews, sentiment ratios, average rating, and negative review count.
3. Review the top negative issue panel.
4. Click `Giao hàng / vận chuyển` to populate the Review Explorer.
5. Ask: `Khách hàng phàn nàn gì về giao hàng chậm?`
6. Show the response sections:
   - Tóm tắt
   - Vấn đề/chủ đề chính
   - Mức độ tin cậy
   - Gợi ý hành động
   - Bằng chứng review

## User Workflows

- Seller checks top complaints before improving operations.
- Analyst explores sentiment by category, product, rating, and keyword.
- Customer-service staff retrieves original evidence reviews before writing a response plan.

## Health Check

```bash
python scripts/check_project_health.py
python scripts/smoke_product_api.py
```

The health check warns if Module 4 corpus/index counts differ from the expected final artifact. It does not rebuild or overwrite artifacts.

## Limitations

- Sentiment metadata in the RAG corpus is inferred and can be noisy.
- Retrieval is an evidence baseline, not a fully reliable business intelligence system.
- Confidence is heuristic, not a calibrated probability.
- Template answers are safer and more transparent than free-form LLM generation, but less fluent.
