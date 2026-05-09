# Error Analysis

## 1. Classification errors

Mô hình sentiment baseline đạt hiệu quả tốt nhưng vẫn chịu ảnh hưởng của mất cân bằng lớp. Lớp neutral/mixed sentiment thường khó hơn vì review có thể vừa khen vừa chê hoặc chỉ mô tả trải nghiệm ở mức trung bình.

- Linear SVM: lớp khó nhất trên test là `neutral` với F1=0.8230.
- Logistic Regression: lớp khó nhất trên test là `neutral` với F1=0.8000.

Các lỗi phân loại còn có thể đến từ nhãn suy luận trên RAG corpus: sentiment của review trong RAG không phải nhãn supervised gốc mà được dự đoán bằng classifier, vì vậy có thể chứa nhiễu.

## 2. Retrieval errors

Semantic retrieval có thể truy hồi tài liệu giống từ khóa nhưng không thật sự liên quan đến ý định câu hỏi. Ví dụ, các review chứa từ khóa về chất lượng, giao hàng hoặc shop có thể xuất hiện trong ngữ cảnh spam, kể chuyện ngoài lề, hoặc nội dung không phải đánh giá sản phẩm.

- File ví dụ truy hồi hiện có: `results\module4\retrieval_eval_examples.jsonl`.

## 3. RAG generation errors

Template-based RAG an toàn hơn vì chỉ tổng hợp từ evidence được truy hồi và có citation, nhưng kém linh hoạt và kém tự nhiên hơn một hệ LLM generation đầy đủ. Module 4 đã bổ sung cơ chế tránh khẳng định quá mức khi bằng chứng tiêu cực yếu.

- Query demo gần nhất: `Khách hàng phàn nàn gì về giao sai hàng, giao chậm hoặc đóng gói bị móp?`.
- Strong negative evidence: `5`.
- Noise filtered count: `0`.

## 4. Data quality issues

- Class imbalance làm majority baseline có vẻ hợp lý về accuracy nhưng yếu về macro-F1.
- Neutral/mixed sentiment khó vì câu chữ không thể hiện rõ thái độ hoặc chứa nhiều polarity trong cùng review.
- Inferred sentiment labels trên RAG corpus có thể noisy.
- Sentiment-rating conflict xuất hiện khi nội dung tiêu cực nhưng rating cao, hoặc ngược lại.
- Review spam, nội dung nhận xu, quảng cáo, hoặc unrelated text làm giảm chất lượng retrieval và RAG.
- Module 3 có `100` review tiêu cực nổi bật để kiểm tra thủ công.

## 5. Mitigation strategies

- Dùng macro-F1 và per-class metrics thay vì chỉ accuracy.
- Tăng dữ liệu hoặc fine-tune PhoBERT/XLM-R để xử lý neutral/mixed tốt hơn.
- Gắn cờ sentiment-rating conflict để ưu tiên kiểm tra thủ công.
- Mở rộng spam/noise filtering cho nội dung nhận xu, quảng cáo, hoặc text không phải review.
- Kết hợp retrieval score với bộ lọc category/sentiment/aspect để giảm tài liệu keyword-similar nhưng irrelevant.
- Với RAG, giữ cơ chế citation và câu trả lời thận trọng khi bằng chứng yếu.