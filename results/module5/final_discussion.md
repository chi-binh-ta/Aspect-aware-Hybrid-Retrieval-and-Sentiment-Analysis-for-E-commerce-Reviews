# Discussion

Pipeline hiện tại có ưu điểm là minh bạch và dễ tái lập. TF-IDF baseline giúp thiết lập mốc hiệu năng rõ ràng trước khi dùng transformer. Retrieval layer bổ sung bằng chứng trực tiếp từ review, nhờ đó phần phân tích không chỉ dựa trên nhãn sentiment tổng hợp.

Hạn chế chính là chưa fine-tune đầy đủ PhoBERT hoặc XLM-R, nhãn sentiment trên RAG corpus là nhãn suy luận, và manual retrieval evaluation còn phụ thuộc vào việc người dùng điền nhãn liên quan thủ công. Template RAG cũng chưa phải một mô hình sinh ngôn ngữ đầy đủ, nhưng phù hợp với mục tiêu tránh unsupported claims.

Hướng phát triển gồm fine-tune PhoBERT/XLM-R, trích xuất aspect tốt hơn, mở rộng human evaluation, xây Streamlit demo, và cải thiện bộ lọc spam/noise cho review nhận xu hoặc nội dung không liên quan.
