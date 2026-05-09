# Conclusion

Đồ án đã xây dựng được hệ thống phân tích cảm xúc và sinh insight dựa trên bằng chứng cho review thương mại điện tử tiếng Việt. Các module xử lý dữ liệu, huấn luyện baseline, suy luận sentiment, xây retrieval index, và tổng hợp báo cáo đã tạo thành một pipeline hoàn chỉnh.

Điểm quan trọng của hệ thống là tách rõ vai trò: classifier dự đoán sentiment, retrieval cung cấp evidence, và RAG template tạo câu trả lời có citation. Khi bằng chứng tiêu cực chưa đủ mạnh, hệ thống tránh kết luận quá mức, giúp kết quả phù hợp hơn với yêu cầu phân tích có căn cứ.
