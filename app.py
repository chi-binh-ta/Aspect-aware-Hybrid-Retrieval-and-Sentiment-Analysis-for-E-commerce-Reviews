from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Project path setup
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from module6_aspect_sentiment import summarize_aspect_evidence  # noqa: E402
from module6_hybrid_retrieval import (  # noqa: E402
    BM25ReviewRetriever,
    Module4DenseRetriever,
    hybrid_retrieve,
)
from module6_query_intent import infer_query_intent  # noqa: E402
from module6_reranker import CrossEncoderReranker  # noqa: E402

# -----------------------------------------------------------------------------
# UI config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Sentiment RAG Demo",
    page_icon="🛒",
    layout="wide",
)

ASPECT_OPTIONS = {
    "Tự động / không lọc": None,
    "Đóng gói": "dong_goi",
    "Giao hàng": "giao_hang",
    "Chất lượng": "chat_luong",
    "Giá": "gia",
    "Dịch vụ shop": "dich_vu_shop",
    "Mẫu mã / size / màu": "mau_ma_size_mau",
}

SAMPLE_QUERIES = [
    "Khách phàn nàn gì về đóng gói?",
    "Review nào nói giao hàng chậm?",
    "Sản phẩm có vấn đề gì về chất lượng?",
    "Khách đánh giá giá cả có đáng tiền không?",
    "Shop có phản hồi hoặc hỗ trợ khách tốt không?",
]

# -----------------------------------------------------------------------------
# Data and model helpers
# -----------------------------------------------------------------------------
def _candidate_corpus_paths(project_dir: Path) -> list[Path]:
    return [
        project_dir / "data" / "processed" / "rag_corpus_module4.csv",
        project_dir / "data" / "processed" / "rag_corpus.csv",
    ]


@st.cache_data(show_spinner=False)
def load_review_corpus(project_dir_str: str) -> tuple[list[dict[str, Any]], str]:
    project_dir = Path(project_dir_str)
    input_path = next((path for path in _candidate_corpus_paths(project_dir) if path.exists()), None)
    if input_path is None:
        expected = " hoặc ".join(str(path) for path in _candidate_corpus_paths(project_dir))
        raise FileNotFoundError(f"Không tìm thấy review corpus. Cần có: {expected}")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Corpus rỗng: {input_path}")

    if "review_id" not in df.columns and "doc_id" in df.columns:
        df["review_id"] = df["doc_id"]
    if "review_id" not in df.columns:
        df["review_id"] = [f"review_{i}" for i in range(len(df))]

    if "text" not in df.columns:
        if "comment" in df.columns:
            df["text"] = df["comment"]
        elif "retrieval_text" in df.columns:
            df["text"] = df["retrieval_text"]
        elif "review" in df.columns:
            df["text"] = df["review"]
        else:
            raise KeyError("Corpus phải có một trong các cột: text, comment, retrieval_text, review")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    return df.to_dict(orient="records"), str(input_path)


@st.cache_resource(show_spinner=False)
def build_bm25_retriever(reviews_json: str) -> BM25ReviewRetriever:
    reviews = json.loads(reviews_json)
    return BM25ReviewRetriever(reviews)


@st.cache_resource(show_spinner=False)
def build_dense_retriever(project_dir_str: str, enabled: bool):
    if not enabled:
        return None, "Dense retriever đã tắt. UI chạy BM25 + query expansion."
    try:
        retriever = Module4DenseRetriever(project_dir=Path(project_dir_str))
        return retriever, "Dense retriever đã sẵn sàng."
    except Exception as exc:  # defensive fallback for missing FAISS/model artifacts
        return None, f"Không load được dense retriever, tự fallback sang BM25. Lý do: {exc}"


@st.cache_resource(show_spinner=False)
def build_reranker(use_cross_encoder: bool, model_name: str):
    if not use_cross_encoder:
        return CrossEncoderReranker(model_name="__force_fallback__")
    return CrossEncoderReranker(model_name=model_name.strip() or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


# -----------------------------------------------------------------------------
# Result processing helpers
# -----------------------------------------------------------------------------
def normalize_evidence_text(text: object) -> str:
    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def score_for_display(result: dict[str, Any]) -> float:
    return coerce_float(result.get("final_score", result.get("rrf_score", 0.0)))


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_text: dict[str, dict[str, Any]] = {}
    first_position: dict[str, int] = {}

    for position, result in enumerate(results):
        key = normalize_evidence_text(result.get("text") or result.get("comment") or result.get("retrieval_text"))
        if not key:
            key = str(result.get("review_id") or result.get("doc_id") or position)
        first_position.setdefault(key, position)
        current = best_by_text.get(key)
        if current is None or score_for_display(result) > score_for_display(current):
            best_by_text[key] = result

    deduped = list(best_by_text.items())
    deduped.sort(key=lambda item: (-score_for_display(item[1]), first_position[item[0]]))
    return [result for _key, result in deduped]


def filter_by_aspect(results: list[dict[str, Any]], target_aspect: str | None) -> list[dict[str, Any]]:
    if not target_aspect:
        return results
    filtered = [
        result
        for result in results
        if target_aspect in result.get("detected_aspects", [])
        or target_aspect in (result.get("aspect_sentiments") or {})
    ]
    return filtered or results


def select_top_evidence(
    query: str,
    results: list[dict[str, Any]],
    target_aspect: str | None,
    top_evidence: int,
) -> list[dict[str, Any]]:
    top_evidence = max(int(top_evidence), 0)
    if top_evidence == 0:
        return []

    results = deduplicate_results(results)
    intent = infer_query_intent(query)

    if not (intent.get("is_complaint_query") and target_aspect):
        return results[:top_evidence]

    complaint_results: list[dict[str, Any]] = []
    other_results: list[dict[str, Any]] = []

    for result in results:
        aspect_sentiments = result.get("aspect_sentiments") or {}
        complaint_score = coerce_float(result.get("complaint_intent_score"))
        if complaint_score > 0 or aspect_sentiments.get(target_aspect) == "negative":
            complaint_results.append(result)
        else:
            other_results.append(result)

    if not complaint_results:
        return results[:top_evidence]

    top_results = complaint_results[:top_evidence]
    if len(top_results) < top_evidence:
        top_results.extend(other_results[: top_evidence - len(top_results)])
    return top_results


def sentiment_badge(sentiment: str) -> str:
    if sentiment == "positive":
        return "🟢 positive"
    if sentiment == "negative":
        return "🔴 negative"
    if sentiment == "mixed":
        return "🟠 mixed"
    return "⚪ neutral"


def short_text(text: object, limit: int = 420) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.title("🛒 Aspect-aware Hybrid Retrieval & Sentiment Demo")
st.caption("Giao diện thử nghiệm cho evidence retrieval, aspect detection và sentiment summary trên review thương mại điện tử tiếng Việt.")

with st.sidebar:
    st.header("Cấu hình truy vấn")
    selected_sample = st.selectbox("Query mẫu", ["Tự nhập"] + SAMPLE_QUERIES)

    target_label = st.selectbox("Aspect cần ưu tiên", list(ASPECT_OPTIONS.keys()), index=1)
    target_aspect = ASPECT_OPTIONS[target_label]

    top_candidates = st.slider("Số candidate ban đầu", min_value=10, max_value=200, value=50, step=10)
    top_evidence = st.slider("Số evidence hiển thị", min_value=3, max_value=20, value=5, step=1)

    st.divider()
    st.subheader("Retrieval / Rerank")
    enable_dense = st.checkbox("Bật dense FAISS nếu artifact tồn tại", value=True)
    use_reranker = st.checkbox("Bật reranker", value=True)
    use_cross_encoder = st.checkbox("Dùng CrossEncoder thật", value=False)
    cross_encoder_model = st.text_input(
        "CrossEncoder model",
        value="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        disabled=not use_cross_encoder,
    )

    st.caption(
        "Mặc định UI dùng lexical fallback để chạy ổn định. Bật CrossEncoder chỉ khi máy đã có mạng/cache model."
    )

try:
    reviews, corpus_path = load_review_corpus(str(PROJECT_DIR))
    reviews_json = json.dumps(reviews, ensure_ascii=False, default=str)
    bm25_retriever = build_bm25_retriever(reviews_json)
    dense_retriever, dense_status = build_dense_retriever(str(PROJECT_DIR), enable_dense)
except Exception as exc:
    st.error(f"Không khởi tạo được dữ liệu: {exc}")
    st.stop()

query_default = selected_sample if selected_sample != "Tự nhập" else "Khách phàn nàn gì về đóng gói?"
query = st.text_area("Nhập câu hỏi kiểm tra", value=query_default, height=90)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Corpus", f"{len(reviews):,} reviews")
col_b.metric("Top candidates", top_candidates)
col_c.metric("Top evidence", top_evidence)

with st.expander("Trạng thái hệ thống", expanded=False):
    st.write(f"Corpus path: `{corpus_path}`")
    st.write(dense_status)
    st.write(f"Target aspect: `{target_aspect or 'none'}`")

run_button = st.button("🔎 Chạy retrieval", type="primary", use_container_width=True)

if run_button:
    if not query.strip():
        st.warning("Bạn cần nhập query trước.")
        st.stop()

    with st.spinner("Đang retrieve và rerank evidence..."):
        candidates = hybrid_retrieve(
            query=query,
            reviews=reviews,
            dense_retriever=dense_retriever,
            bm25_top_k=top_candidates,
            dense_top_k=top_candidates,
            final_top_k=top_candidates,
            bm25_retriever=bm25_retriever,
        )

        if use_reranker:
            reranker = build_reranker(use_cross_encoder, cross_encoder_model)
            ranked = reranker.rerank(query, candidates, top_k=top_candidates)
        else:
            ranked = candidates

        ranked = filter_by_aspect(ranked, target_aspect)
        top_results = select_top_evidence(query, ranked, target_aspect, top_evidence)
        summary = summarize_aspect_evidence(top_results)

    st.success(f"Tìm thấy {len(top_results)} evidence sau lọc/rerank.")

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Top evidence")
        if not top_results:
            st.info("Không tìm thấy evidence phù hợp.")
        for idx, result in enumerate(top_results, start=1):
            aspect_sentiments = result.get("aspect_sentiments") or {}
            detected = result.get("detected_aspects") or []
            score = score_for_display(result)
            source_flags = ", ".join(result.get("source_flags", [])) or "reranker"
            sentiment = "mixed" if len(set(aspect_sentiments.values())) > 1 else next(iter(aspect_sentiments.values()), "neutral")

            with st.container(border=True):
                h1, h2, h3 = st.columns([0.7, 1, 1])
                h1.markdown(f"**#{idx}**")
                h2.markdown(f"Score: `{score:.4f}`")
                h3.markdown(sentiment_badge(sentiment))

                st.write(short_text(result.get("text")))

                meta_cols = st.columns(3)
                meta_cols[0].caption(f"review_id: {result.get('review_id', 'n/a')}")
                meta_cols[1].caption(f"source: {source_flags}")
                meta_cols[2].caption(f"aspects: {', '.join(detected) if detected else 'none'}")

                if aspect_sentiments:
                    st.json(aspect_sentiments, expanded=False)

    with right:
        st.subheader("Aspect summary")
        display_summary = summary
        if target_aspect and target_aspect in summary:
            display_summary = {target_aspect: summary[target_aspect]}

        if not display_summary:
            st.info("Chưa có aspect summary.")
        else:
            for aspect, payload in display_summary.items():
                with st.container(border=True):
                    st.markdown(f"**{aspect}** — {sentiment_badge(payload.get('sentiment', 'neutral'))}")
                    evidence_items = payload.get("evidence", [])[:3]
                    for ev in evidence_items:
                        st.caption(f"• {short_text(ev.get('text'), limit=160)}")

        payload = {
            "query": query,
            "target_aspect": target_aspect,
            "results": top_results,
            "aspect_summary": summary,
        }
        st.download_button(
            "⬇️ Tải kết quả JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            file_name="module6_ui_result.json",
            mime="application/json",
            use_container_width=True,
        )
else:
    st.info("Nhập query rồi bấm **Chạy retrieval** để kiểm tra pipeline.")
