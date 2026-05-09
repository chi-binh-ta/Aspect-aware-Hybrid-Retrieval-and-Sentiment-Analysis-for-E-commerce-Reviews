"""Generate Vietnamese final report sections for Module 5."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def resolve_project_dir() -> Path:
    env_dir = os.environ.get("PROJECT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def load_summary(project_dir: Path) -> dict:
    path = project_dir / "results/module5/final_project_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, default: str = "chưa có") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    project_dir = resolve_project_dir()
    report_dir = project_dir / "docs" / "final_report"
    results_dir = project_dir / "results" / "module5"
    report_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(project_dir)
    best_model = fmt(summary.get("module2_best_model"))
    best_f1 = fmt(summary.get("module2_best_macro_f1"))
    module4_query = fmt(summary.get("module4_rag_query"))
    strong_negative = fmt(summary.get("module4_strong_negative_count"))

    skeleton = """# Final Report Skeleton

## 1. Introduction

## 2. Related Work

## 3. Dataset

## 4. Methodology

## 5. Experiments

## 6. Error Analysis

## 7. Demo System

## 8. Conclusion
"""

    sections = f"""# Các phần nội dung cho báo cáo cuối kỳ

## 1. Introduction

Đề tài xây dựng một hệ thống phân tích cảm xúc đánh giá sản phẩm thương mại điện tử tiếng Việt kết hợp với tầng truy hồi bằng chứng. Hệ thống gồm hai năng lực chính: mô hình phân loại sentiment dự đoán nhãn negative, neutral hoặc positive cho review; và retrieval/RAG cung cấp các review liên quan để hỗ trợ sinh nhận định có căn cứ.

## 3. Dataset

Dữ liệu sentiment được chuẩn hóa từ file đánh giá có nhãn rating, sau đó ánh xạ thành ba lớp cảm xúc. Corpus RAG được xây dựng từ các review sản phẩm thuộc nhiều danh mục, giữ metadata như category, product, rating và nội dung review. Sentiment trên RAG corpus là nhãn suy luận từ classifier, không phải nhãn gốc được gán thủ công.

## 4. Methodology

Module 2 huấn luyện các baseline cổ điển TF-IDF + Logistic Regression và TF-IDF + Linear SVM. Mô hình tốt nhất được chọn bằng validation macro-F1, sau đó báo cáo trên test set. Trong project hiện tại, best model là `{best_model}` với macro-F1 khoảng `{best_f1}`.

Module 4 xây dựng retrieval layer bằng multilingual sentence embedding và FAISS. Query được nhúng theo format E5, retrieval trả về review kèm metadata và sentiment. Tầng RAG không gọi API LLM mà dùng template tổng hợp có citation, giúp kiểm soát tốt hơn việc tránh khẳng định không có bằng chứng.

## 5. Experiments

Kết quả sentiment classification nên được trình bày bằng accuracy, macro-F1, weighted-F1 và per-class F1. Với retrieval, hệ thống cung cấp sheet đánh giá thủ công Precision@k; chỉ tính metric khi cột `is_relevant_manual` đã được điền 0/1.

## 6. Error Analysis

Các lỗi chính gồm mất cân bằng lớp, khó phân biệt neutral/mixed sentiment, nhãn sentiment suy luận có nhiễu, xung đột sentiment-rating, spam review hoặc nội dung nhận xu. Với retrieval, hệ thống có thể lấy tài liệu giống keyword nhưng không đúng intent. Với RAG, template-based generation an toàn hơn nhưng kém linh hoạt hơn full LLM.

## 7. Demo System

Ví dụ query RAG gần nhất: `{module4_query}`. Module 4 ghi nhận `{strong_negative}` bằng chứng tiêu cực mạnh trong lần chạy demo gần nhất. Khi bằng chứng yếu, hệ thống dùng câu trả lời thận trọng thay vì khẳng định khách hàng phàn nàn.

## 8. Conclusion

Hệ thống chứng minh pipeline NLP/RAG end-to-end cho bài toán review insight: classifier dự đoán sentiment, retrieval cung cấp bằng chứng, và RAG tạo nhận định có citation. Các kết quả phù hợp cho baseline cuối kỳ, đồng thời vẫn còn không gian cải thiện bằng mô hình ngôn ngữ mạnh hơn và đánh giá thủ công lớn hơn.
"""

    discussion = """# Discussion

Pipeline hiện tại có ưu điểm là minh bạch và dễ tái lập. TF-IDF baseline giúp thiết lập mốc hiệu năng rõ ràng trước khi dùng transformer. Retrieval layer bổ sung bằng chứng trực tiếp từ review, nhờ đó phần phân tích không chỉ dựa trên nhãn sentiment tổng hợp.

Hạn chế chính là chưa fine-tune đầy đủ PhoBERT hoặc XLM-R, nhãn sentiment trên RAG corpus là nhãn suy luận, và manual retrieval evaluation còn phụ thuộc vào việc người dùng điền nhãn liên quan thủ công. Template RAG cũng chưa phải một mô hình sinh ngôn ngữ đầy đủ, nhưng phù hợp với mục tiêu tránh unsupported claims.

Hướng phát triển gồm fine-tune PhoBERT/XLM-R, trích xuất aspect tốt hơn, mở rộng human evaluation, xây Streamlit demo, và cải thiện bộ lọc spam/noise cho review nhận xu hoặc nội dung không liên quan.
"""

    conclusion = """# Conclusion

Đồ án đã xây dựng được hệ thống phân tích cảm xúc và sinh insight dựa trên bằng chứng cho review thương mại điện tử tiếng Việt. Các module xử lý dữ liệu, huấn luyện baseline, suy luận sentiment, xây retrieval index, và tổng hợp báo cáo đã tạo thành một pipeline hoàn chỉnh.

Điểm quan trọng của hệ thống là tách rõ vai trò: classifier dự đoán sentiment, retrieval cung cấp evidence, và RAG template tạo câu trả lời có citation. Khi bằng chứng tiêu cực chưa đủ mạnh, hệ thống tránh kết luận quá mức, giúp kết quả phù hợp hơn với yêu cầu phân tích có căn cứ.
"""

    outputs = {
        report_dir / "report_skeleton.md": skeleton,
        report_dir / "final_report_sections.md": sections,
        results_dir / "final_discussion.md": discussion,
        results_dir / "final_conclusion.md": conclusion,
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(f"[module5_report] Saved: {path}")


if __name__ == "__main__":
    main()
