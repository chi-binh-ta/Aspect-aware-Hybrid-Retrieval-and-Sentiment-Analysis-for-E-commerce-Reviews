# Module 4 RAG Demo Output

**Query:** Khách hàng phàn nàn gì về giao sai hàng, giao chậm hoặc đóng gói bị móp?

## Answer

Dựa trên 8 review được truy hồi, có 5 bằng chứng tiêu cực rõ ràng.
Các phàn nàn chính liên quan đến: dịch vụ/shop (20), mẫu mã/size/màu (13), giá (5), giao hàng (3).
Bằng chứng chính nằm ở các review [1] [2] [3].
Các phản hồi tiêu cực được ưu tiên trích dẫn khi sentiment là negative và rating thấp hoặc có dấu hiệu khiếu nại trong nội dung.
Trích dẫn bằng chứng:
[1] Lần đầu mua hàng tại shop mình đã có trải nghiệm thật sự rất khó chịu. Giao hàng rất nhanh tuy nhiên lần đầu mình mua trắng nhưng shop giao màu đen, 1 phần do mình đã có dự định về quần áo định mặc nên kh thể dùng luôn màu đen đc mình có nt cho shop để xem... (sentiment=negative, category=Fashion, rating=1)
[2] Shop giao sai hàng nhắn tin rất nhiều lần không trả lời , điện thoại 10 c không nghe máy. để shop giải quyết trước nhưng ko dc, phải bấm hoàn tiền trên shop Pi ,nhắn tin rất nhiều lần thì shop mới trả lời một câu “ok đã bấm hoàn tiền” Hết.,Trong khi sóc là... (sentiment=negative, category=Electronic, rating=3)
[3] Chất liệu:ok Đúng với mô tả:ok Màu sắc:ok In ảnh cho nó có thể là do mộtmột phần mềm khác thì cũng phải chi nó bảo ko phải có cái gì này thì có lẽ là do các bạn ấy có vẻ là mlà ngườngười mẫu ik học sinh lớp cô hà Hồ thì ai mà biết nó có cái lý do của họ là... (sentiment=negative, category=Fashion, rating=5)

## Evidence

- [1] `rag_Fashion_12183625609` | score=0.8294147253036499 | sentiment=negative | Lần đầu mua hàng tại shop mình đã có trải nghiệm thật sự rất khó chịu. Giao hàng rất nhanh tuy nhiên lần đầu mình mua trắng nhưng shop giao màu đen, 1 phần do mình đã có dự định về quần áo định mặc nên kh thể dùng luôn màu đen đc mình có nt cho shop để xem...
- [2] `rag_Electronic_12171819321` | score=0.8266198635101318 | sentiment=negative | Shop giao sai hàng nhắn tin rất nhiều lần không trả lời , điện thoại 10 c không nghe máy. để shop giải quyết trước nhưng ko dc, phải bấm hoàn tiền trên shop Pi ,nhắn tin rất nhiều lần thì shop mới trả lời một câu “ok đã bấm hoàn tiền” Hết.,Trong khi sóc là...
- [3] `rag_Fashion_14213030635` | score=0.8262240886688232 | sentiment=negative | Chất liệu:ok Đúng với mô tả:ok Màu sắc:ok In ảnh cho nó có thể là do mộtmột phần mềm khác thì cũng phải chi nó bảo ko phải có cái gì này thì có lẽ là do các bạn ấy có vẻ là mlà ngườngười mẫu ik học sinh lớp cô hà Hồ thì ai mà biết nó có cái lý do của họ là...
- [4] `rag_Electronic_15741431389` | score=0.825767993927002 | sentiment=negative | Màu xinh mà hơi khó kết nối hôm mua tới giờ chưa kết nối dc luôn k biết lỗi ở đâu .m nghĩ mấy đồ này nên mua ngoài tiệm có ng tới nhà bảo hành vs lắp đặt chứ nói trên đt k giải quyết dc
- [5] `rag_Fashion_13764220307` | score=0.8248525261878967 | sentiment=negative | Màu sắc:sai Đúng với mô tả:đúng Chất liệu:vải hơi mỏng nha Tuy shop còn ship sai màu , mik đặt là màu trắng nhm ra màu đen. Vải mỏng nx nhg vs giá mik săn dc 40k v cx ngon r