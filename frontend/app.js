const API_BASE = localStorage.getItem("API_BASE") || "http://127.0.0.1:8000";

const suggestedQuestions = [
  "Khách hàng phàn nàn gì nhiều nhất?",
  "Review 1 sao thường nói về vấn đề gì?",
  "Khách hàng khen điểm gì?",
  "Có vấn đề nào về giao hàng không?",
  "Có vấn đề nào về đóng gói không?",
  "Sản phẩm có bị chê sai mô tả không?"
];

let reviewOffset = 0;
const reviewLimit = 25;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  })[char]);
}

function fmtNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return Number(value).toLocaleString("vi-VN");
}

function pct(count, total) {
  if (!total) return "0.0%";
  return `${((count / total) * 100).toFixed(1)}%`;
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setStatus(text, ok = true) {
  const status = document.getElementById("api-status");
  status.textContent = text;
  status.style.background = ok ? "rgba(40, 180, 110, 0.22)" : "rgba(220, 70, 55, 0.22)";
}

function renderSummary(summary) {
  const sentiment = summary.sentiment_distribution || {};
  const total = summary.total_reviews || 0;
  const positive = sentiment.positive || 0;
  const neutral = sentiment.neutral || 0;
  const negative = sentiment.negative || 0;
  const cards = [
    ["Total reviews", fmtNumber(total)],
    ["Positive", pct(positive, total)],
    ["Neutral", pct(neutral, total)],
    ["Negative", pct(negative, total)],
    ["Average rating", summary.average_rating ?? "N/A"],
    ["Needs attention", fmtNumber(summary.negative_review_count || 0)]
  ];
  document.getElementById("summary-cards").innerHTML = cards.map(([label, value]) => `
    <div class="metric-card">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>
  `).join("");

  const maxCount = Math.max(positive, neutral, negative, 1);
  document.getElementById("sentiment-bars").innerHTML = ["positive", "neutral", "negative"].map((name) => {
    const count = sentiment[name] || 0;
    const width = Math.max((count / maxCount) * 100, count ? 3 : 0);
    return `
      <div class="bar-row">
        <strong>${name}</strong>
        <div class="bar-track"><div class="bar-fill ${name}" style="width:${width}%"></div></div>
        <span>${pct(count, total)}</span>
      </div>
    `;
  }).join("");

  const warningBox = document.getElementById("warning-box");
  if (summary.warnings && summary.warnings.length) {
    warningBox.classList.remove("hidden");
    warningBox.innerHTML = summary.warnings.map((warning) => `<div>${escapeHtml(warning)}</div>`).join("");
  } else {
    warningBox.classList.add("hidden");
  }

  const categorySelect = document.getElementById("filter-category");
  const categories = summary.available_filters?.categories || [];
  categorySelect.innerHTML = `<option value="">Danh mục</option>` + categories.map((category) => (
    `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`
  )).join("");
}

function renderIssues(data) {
  const container = document.getElementById("issues-list");
  const issues = data.issues || [];
  if (!issues.length) {
    container.innerHTML = `<p class="muted">Chưa tìm thấy issue nổi bật với bộ lọc hiện tại.</p>`;
    return;
  }
  container.innerHTML = issues.map((issue) => {
    const examples = (issue.example_reviews || []).map((item) => (
      `<p class="muted">“${escapeHtml(item.review_text).slice(0, 150)}”</p>`
    )).join("");
    return `
      <article class="issue-card">
        <strong>${escapeHtml(issue.label_vi)}</strong>
        <div class="meta-line">
          <span>${fmtNumber(issue.count)} review</span>
          <span>${(issue.share * 100).toFixed(1)}%</span>
        </div>
        ${examples}
        <button class="secondary-btn issue-search" data-label="${escapeHtml(issue.label_vi)}">Xem review liên quan</button>
      </article>
    `;
  }).join("");
  document.querySelectorAll(".issue-search").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById("search-q").value = button.dataset.label.split("/")[0].trim();
      document.getElementById("filter-sentiment").value = "negative";
      searchReviews(true);
    });
  });
}

function renderQuestionChips() {
  document.getElementById("question-chips").innerHTML = suggestedQuestions.map((question) => (
    `<button class="chip" data-question="${escapeHtml(question)}">${escapeHtml(question)}</button>`
  )).join("");
  document.querySelectorAll(".chip").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById("rag-query").value = button.dataset.question;
      askRag();
    });
  });
}

function renderRag(data) {
  const confidence = data.confidence || "low";
  const evidence = data.evidence || [];
  const themes = data.main_themes || [];
  document.getElementById("rag-output").innerHTML = `
    <div class="meta-line">
      <span class="confidence-badge confidence-${escapeHtml(confidence)}">Confidence: ${escapeHtml(confidence)}</span>
      <span>${escapeHtml(data.evidence_quality?.reason || "")}</span>
    </div>
    <div class="answer-block">${escapeHtml(data.answer || "")}</div>
    <h3>Chủ đề chính</h3>
    <div class="chips">${themes.map((theme) => `<span class="tag">${escapeHtml(theme.label_vi)} (${fmtNumber(theme.count)})</span>`).join("") || "<span class='muted'>Chưa rõ</span>"}</div>
    <h3>Evidence reviews</h3>
    <div class="review-results">
      ${evidence.map((item, idx) => `
        <article class="evidence-card">
          <div class="meta-line">
            <span class="tag ${escapeHtml(item.sentiment || "")}">[${idx + 1}] ${escapeHtml(item.sentiment || "unknown")}</span>
            <span>Rating: ${item.rating ?? "N/A"}</span>
            <span>${escapeHtml(item.category || "")}</span>
            <span>${escapeHtml(item.score_or_match_reason || "")}</span>
          </div>
          <p>${escapeHtml(item.review_text || "")}</p>
        </article>
      `).join("")}
    </div>
    ${(data.warnings || []).map((warning) => `<p class="warning-box">${escapeHtml(warning)}</p>`).join("")}
  `;
}

function renderReviews(data, append = false) {
  const container = document.getElementById("review-results");
  const html = (data.items || []).map((item) => `
    <article class="review-card">
      <div class="meta-line">
        <span class="tag ${escapeHtml(item.sentiment || "")}">${escapeHtml(item.sentiment || "")}</span>
        <span>Rating: ${item.rating ?? "N/A"}</span>
        <span>${escapeHtml(item.category || "")}</span>
        <span>${escapeHtml(item.product_name ? item.product_name.slice(0, 80) : "")}</span>
      </div>
      <p>${escapeHtml(item.review_text)}</p>
      <p class="muted">${escapeHtml(item.score_or_match_reason || "")}</p>
    </article>
  `).join("");
  container.innerHTML = append ? container.innerHTML + html : html;
  document.getElementById("load-more").classList.toggle(
    "hidden",
    reviewOffset + reviewLimit >= (data.total_estimate || 0)
  );
}

async function loadSummary() {
  try {
    const health = await api("/api/health");
    setStatus(`API OK · ${fmtNumber(health.total_reviews)} reviews`, true);
    const summary = await api("/api/analytics/summary");
    renderSummary(summary);
  } catch (error) {
    setStatus(`API unavailable: ${error.message}`, false);
  }
}

async function loadIssues() {
  const data = await api("/api/analytics/issues?sentiment=negative&limit=8");
  renderIssues(data);
}

async function askRag() {
  const query = document.getElementById("rag-query").value.trim();
  if (!query) return;
  const sentiment = document.getElementById("rag-sentiment").value;
  document.getElementById("rag-output").innerHTML = `<p class="muted">Đang truy xuất review bằng chứng...</p>`;
  const data = await api("/api/rag", {
    method: "POST",
    body: JSON.stringify({ query, sentiment: sentiment || null, top_k: 5 })
  });
  renderRag(data);
}

async function searchReviews(reset = true) {
  if (reset) reviewOffset = 0;
  const params = new URLSearchParams();
  const q = document.getElementById("search-q").value.trim();
  const sentiment = document.getElementById("filter-sentiment").value;
  const ratingMin = document.getElementById("filter-rating-min").value;
  const category = document.getElementById("filter-category").value;
  const product = document.getElementById("filter-product").value.trim();
  if (q) params.set("q", q);
  if (sentiment) params.set("sentiment", sentiment);
  if (ratingMin) params.set("rating_min", ratingMin);
  if (category) params.set("category", category);
  if (product) params.set("product", product);
  params.set("limit", reviewLimit);
  params.set("offset", reviewOffset);
  const data = await api(`/api/reviews/search?${params.toString()}`);
  renderReviews(data, !reset);
}

document.getElementById("refresh-issues").addEventListener("click", loadIssues);
document.getElementById("ask-rag").addEventListener("click", askRag);
document.getElementById("search-reviews").addEventListener("click", () => searchReviews(true));
document.getElementById("load-more").addEventListener("click", () => {
  reviewOffset += reviewLimit;
  searchReviews(false);
});

renderQuestionChips();
loadSummary().then(loadIssues).then(() => searchReviews(true));
