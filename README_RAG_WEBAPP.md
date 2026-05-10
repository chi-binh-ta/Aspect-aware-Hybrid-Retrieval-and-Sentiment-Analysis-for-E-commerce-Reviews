# RAG Web App Notes

The web app is a lightweight frontend plus FastAPI backend for the Vietnamese E-commerce Review Intelligence Assistant.

## Backend Endpoints

- `GET /api/health`: API and corpus status.
- `GET /api/analytics/summary`: total reviews, sentiment distribution, rating distribution, average rating, available filters, and warnings.
- `GET /api/analytics/issues`: keyword-based issue taxonomy over filtered reviews.
- `GET /api/reviews/search`: review explorer with keyword, sentiment, rating, category, and product filters.
- `POST /api/rag`: evidence-based RAG-style answer with summary, themes, suggested actions, confidence, evidence quality, warnings, and evidence reviews.

## Frontend Sections

- Dashboard summary cards.
- Sentiment distribution bars.
- Top issues panel.
- Suggested question chips.
- Review Explorer.
- Structured RAG answer with confidence and evidence cards.

## Example Questions

- `Khách hàng phàn nàn gì nhiều nhất?`
- `Review 1 sao thường nói về vấn đề gì?`
- `Khách hàng khen điểm gì?`
- `Có vấn đề nào về giao hàng không?`
- `Có vấn đề nào về đóng gói không?`
- `Sản phẩm có bị chê sai mô tả không?`

## Confidence And Evidence Quality

The app estimates confidence from simple evidence overlap and issue-keyword signals:

- `high`: at least four retrieved reviews look directly related.
- `medium`: two or three retrieved reviews look related.
- `low`: fewer than two related evidence reviews or weak retrieval evidence.

This is transparent triage logic, not a calibrated probability.

## Artifact Safety

Demo scripts reuse the existing Module 4 corpus and FAISS index. Rebuilds are intentionally isolated in `scripts/rebuild_module4_index.sh`.

## Limitations

- Retrieval quality is moderate and depends on the active corpus/index.
- Some review text contains spam, emoji, noisy abbreviations, or unrelated content.
- The answer generator is template-based and evidence-centric; it is not a full LLM generation system.
