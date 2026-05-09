"""Generate final error analysis markdown for the project."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def resolve_project_dir() -> Path:
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def class_report_notes(project_dir: Path) -> list[str]:
    notes = []
    for name, rel in [
        ("Linear SVM", "results/module2/linear_svm_classification_report.csv"),
        ("Logistic Regression", "results/module2/logistic_regression_classification_report.csv"),
    ]:
        df = read_csv(project_dir / rel)
        if df is None or df.empty or not {"split", "label", "f1_score"}.issubset(df.columns):
            notes.append(f"- Không tìm thấy classification report cho {name}.")
            continue
        test = df[df["split"] == "test"].copy()
        if test.empty:
            notes.append(f"- {name}: chưa có dòng test trong classification report.")
            continue
        class_rows = test[test["label"].isin(["negative", "neutral", "positive"])].copy()
        if class_rows.empty:
            continue
        weakest = class_rows.sort_values("f1_score").iloc[0]
        notes.append(
            f"- {name}: lớp khó nhất trên test là `{weakest['label']}` "
            f"với F1={float(weakest['f1_score']):.4f}."
        )
    return notes


def main() -> None:
    project_dir = resolve_project_dir()
    results_dir = project_dir / "results" / "module5"
    results_dir.mkdir(parents=True, exist_ok=True)

    module3_top_negative = read_csv(project_dir / "results/module3/top_negative_reviews.csv")
    rag_output = read_json(project_dir / "results/module4/rag_outputs.json")
    retrieval_examples = project_dir / "results/module4/retrieval_eval_examples.jsonl"

    lines = [
        "# Error Analysis",
        "",
        "## 1. Classification errors",
        "",
        "Mô hình sentiment baseline đạt hiệu quả tốt nhưng vẫn chịu ảnh hưởng của mất cân bằng lớp. "
        "Lớp neutral/mixed sentiment thường khó hơn vì review có thể vừa khen vừa chê hoặc chỉ mô tả trải nghiệm ở mức trung bình.",
        "",
        *class_report_notes(project_dir),
        "",
        "Các lỗi phân loại còn có thể đến từ nhãn suy luận trên RAG corpus: sentiment của review trong RAG không phải nhãn supervised gốc mà được dự đoán bằng classifier, vì vậy có thể chứa nhiễu.",
        "",
        "## 2. Retrieval errors",
        "",
        "Semantic retrieval có thể truy hồi tài liệu giống từ khóa nhưng không thật sự liên quan đến ý định câu hỏi. "
        "Ví dụ, các review chứa từ khóa về chất lượng, giao hàng hoặc shop có thể xuất hiện trong ngữ cảnh spam, kể chuyện ngoài lề, hoặc nội dung không phải đánh giá sản phẩm.",
        "",
    ]
    if retrieval_examples.exists():
        lines.append(f"- File ví dụ truy hồi hiện có: `{retrieval_examples.relative_to(project_dir)}`.")
    else:
        lines.append("- Chưa có file ví dụ truy hồi từ Module 4.")

    lines.extend(
        [
            "",
            "## 3. RAG generation errors",
            "",
            "Template-based RAG an toàn hơn vì chỉ tổng hợp từ evidence được truy hồi và có citation, nhưng kém linh hoạt và kém tự nhiên hơn một hệ LLM generation đầy đủ. "
            "Module 4 đã bổ sung cơ chế tránh khẳng định quá mức khi bằng chứng tiêu cực yếu.",
            "",
        ]
    )
    if rag_output:
        lines.append(f"- Query demo gần nhất: `{rag_output.get('query')}`.")
        lines.append(f"- Strong negative evidence: `{rag_output.get('strong_negative_count')}`.")
        lines.append(f"- Noise filtered count: `{rag_output.get('noise_filtered_count')}`.")
    else:
        lines.append("- Không tìm thấy `results/module4/rag_outputs.json`.")

    lines.extend(
        [
            "",
            "## 4. Data quality issues",
            "",
            "- Class imbalance làm majority baseline có vẻ hợp lý về accuracy nhưng yếu về macro-F1.",
            "- Neutral/mixed sentiment khó vì câu chữ không thể hiện rõ thái độ hoặc chứa nhiều polarity trong cùng review.",
            "- Inferred sentiment labels trên RAG corpus có thể noisy.",
            "- Sentiment-rating conflict xuất hiện khi nội dung tiêu cực nhưng rating cao, hoặc ngược lại.",
            "- Review spam, nội dung nhận xu, quảng cáo, hoặc unrelated text làm giảm chất lượng retrieval và RAG.",
        ]
    )
    if module3_top_negative is not None and not module3_top_negative.empty:
        lines.append(f"- Module 3 có `{len(module3_top_negative)}` review tiêu cực nổi bật để kiểm tra thủ công.")

    lines.extend(
        [
            "",
            "## 5. Mitigation strategies",
            "",
            "- Dùng macro-F1 và per-class metrics thay vì chỉ accuracy.",
            "- Tăng dữ liệu hoặc fine-tune PhoBERT/XLM-R để xử lý neutral/mixed tốt hơn.",
            "- Gắn cờ sentiment-rating conflict để ưu tiên kiểm tra thủ công.",
            "- Mở rộng spam/noise filtering cho nội dung nhận xu, quảng cáo, hoặc text không phải review.",
            "- Kết hợp retrieval score với bộ lọc category/sentiment/aspect để giảm tài liệu keyword-similar nhưng irrelevant.",
            "- Với RAG, giữ cơ chế citation và câu trả lời thận trọng khi bằng chứng yếu.",
        ]
    )

    output_path = results_dir / "error_analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[module5_error] Saved: {output_path}")


if __name__ == "__main__":
    main()

