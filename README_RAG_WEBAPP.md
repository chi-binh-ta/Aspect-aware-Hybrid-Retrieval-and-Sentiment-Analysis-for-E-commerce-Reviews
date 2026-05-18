# RAG Web App Notes

The web app is a lightweight FastAPI backend plus static HTML/CSS/JS frontend for the Vietnamese E-commerce Review Intelligence demo.

This is a local inspection/demo layer over existing artifacts. It is not Streamlit, not a production deployment, and not a full LLM generation system.

## Run

```bash
python scripts/run_rag_webapp.py
```

Default URL:

```text
http://127.0.0.1:8000
```

## Backend Endpoints

- `GET /`: serves `frontend/index.html`.
- `GET /api/health`: API and corpus status.
- `GET /api/config`: demo configuration, available filters, and model availability.
- `POST /api/sentiment`: predicts sentiment for one text if the Module 2 model is available.
- `GET /api/analytics/summary`: review count, sentiment distribution, rating distribution, average rating, and filters.
- `GET /api/analytics/issues`: lightweight keyword-based issue taxonomy over filtered reviews.
- `GET /api/reviews/search`: review explorer with keyword, sentiment, rating, category, and product filters.
- `POST /api/rag`: evidence-based retrieval answer with summary, themes, suggested actions, confidence, evidence quality, warnings, and review evidence.

## Frontend Sections

- Dashboard summary cards.
- Sentiment distribution bars.
- Top issues panel.
- Suggested question chips.
- Review Explorer.
- Structured evidence answer with confidence and evidence cards.

## Example Questions

- `Khách hàng phàn nàn gì nhiều nhất?`
- `Review 1 sao thường nói về vấn đề gì?`
- `Khách hàng khen điểm gì?`
- `Có vấn đề nào về giao hàng không?`
- `Có vấn đề nào về đóng gói không?`
- `Sản phẩm có bị chê sai mô tả không?`

## Artifact Behavior

The backend reads existing local artifacts. If dense retrieval dependencies or FAISS artifacts are unavailable, `/api/rag` falls back to keyword evidence and returns a warning instead of crashing the whole app.

## Limitations

- Retrieval quality depends on the active corpus and index.
- Sentiment labels on the RAG corpus are inferred and may be noisy.
- Confidence is heuristic, not a calibrated probability.
- The answer generator is template-based and evidence-centric, not a full LLM answer generator.
