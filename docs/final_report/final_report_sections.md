# Các phần nội dung cho báo cáo cuối kỳ

## 1. Introduction

Đề tài xây dựng một hệ thống phân tích cảm xúc đánh giá sản phẩm thương mại điện tử tiếng Việt kết hợp với tầng truy hồi bằng chứng. Hệ thống gồm hai năng lực chính: mô hình phân loại sentiment dự đoán nhãn negative, neutral hoặc positive cho review; và retrieval/RAG cung cấp các review liên quan để hỗ trợ sinh nhận định có căn cứ.

## 3. Dataset

Dữ liệu sentiment được chuẩn hóa từ file đánh giá có nhãn rating, sau đó ánh xạ thành ba lớp cảm xúc. Corpus RAG được xây dựng từ các review sản phẩm thuộc nhiều danh mục, giữ metadata như category, product, rating và nội dung review. Sentiment trên RAG corpus là nhãn suy luận từ classifier, không phải nhãn gốc được gán thủ công.

## 4. Methodology

Module 2 huấn luyện các baseline cổ điển TF-IDF + Logistic Regression và TF-IDF + Linear SVM. Mô hình tốt nhất được chọn bằng validation macro-F1, sau đó báo cáo trên test set. Trong project hiện tại, best model là `linear_svm` với macro-F1 khoảng `0.9098`.

Module 4 xây dựng retrieval layer bằng multilingual sentence embedding và FAISS. Query được nhúng theo format E5, retrieval trả về review kèm metadata và sentiment. Tầng RAG không gọi API LLM mà dùng template tổng hợp có citation, giúp kiểm soát tốt hơn việc tránh khẳng định không có bằng chứng.

## 5. Experiments

Kết quả sentiment classification nên được trình bày bằng accuracy, macro-F1, weighted-F1 và per-class F1. Với retrieval, hệ thống cung cấp sheet đánh giá thủ công Precision@k; chỉ tính metric khi cột `is_relevant_manual` đã được điền 0/1.

## 6. Error Analysis

Các lỗi chính gồm mất cân bằng lớp, khó phân biệt neutral/mixed sentiment, nhãn sentiment suy luận có nhiễu, xung đột sentiment-rating, spam review hoặc nội dung nhận xu. Với retrieval, hệ thống có thể lấy tài liệu giống keyword nhưng không đúng intent. Với RAG, template-based generation an toàn hơn nhưng kém linh hoạt hơn full LLM.

## 7. Demo System

Ví dụ query RAG gần nhất: `Khách hàng phàn nàn gì về giao sai hàng, giao chậm hoặc đóng gói bị móp?`. Module 4 ghi nhận `5` bằng chứng tiêu cực mạnh trong lần chạy demo gần nhất. Khi bằng chứng yếu, hệ thống dùng câu trả lời thận trọng thay vì khẳng định khách hàng phàn nàn.

## 8. Conclusion

Hệ thống chứng minh pipeline NLP/RAG end-to-end cho bài toán review insight: classifier dự đoán sentiment, retrieval cung cấp bằng chứng, và RAG tạo nhận định có citation. Các kết quả phù hợp cho baseline cuối kỳ, đồng thời vẫn còn không gian cải thiện bằng mô hình ngôn ngữ mạnh hơn và đánh giá thủ công lớn hơn.
