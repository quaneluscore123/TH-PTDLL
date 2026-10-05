-- Đối với bài tập
Cứ phân tích hết 82 triệu dòng -- chạy theo dung lượng tối đa của máy có thể chạy 5% dữ liệu, 10% tuỳ vào RAM của máy
Dự đoán theo ngày cho đến ngày bay gồm nhiều yếu tố như là lạm phát và các yếu tố khác, trước hết thì tính toán hệ số tương quan để xem nên đưa chỉ số nào vào để huấn luyện.
Dùng Spark để tính toán.
Hãy chỉnh sửa sao cho thuật toán tính toán dữ liệu sát nhất với năm 2023.
Dựa vào dữ liệu hiện có 2022 trong máy thì phân tích dự đoán cho 2023 rồi so với dữ liệu thực tế, tiếp theo là dự đoán cho đến 2026 rồi đến tương lai 2027.

-- Đối với báo cáo
notebook, dashboard, slide
So sánh nhiều mô hình
Đưa ra dữ liệu đã làm ra

bạn có thể phân tích giá vé từ tháng 4 đến tháng 8 của dữ liệu dự đoán từ tháng 9 đến tháng 10 để kiểm tra xem mô hình dự đoán và phân tích có oke hay không rồi từ đó có thể phân tích đến các năm các tháng tiếp theo. "Phân tích hết 82 triệu dòng". Tôi hiểu là: các bảng thống kê và tương quan chạy trên toàn bộ 82 triệu dòng, còn huấn luyện và so sánh mô hình chạy trên mẫu 5–10%. Hiểu như vậy có đúng không? đúng, Ổ D chỉ còn 11,7 GB. Chuyển toàn bộ 82 triệu dòng sang Parquet cần khoảng 6–10 GB, cộng thêm file tạm của Spark, nên sẽ thiếu chỗ. Bạn có thể dọn thêm ổ D không? Hoặc cho phép tôi dùng ổ C (còn 43 GB) cho file tạm và tầng Bronze?
làm ở ổ E, Dashboard làm bằng công cụ gì? Streamlit (Python, tôi làm trọn được), Power BI (bạn tự kéo thả từ bảng Gold), hay một trang HTML tương tác? HTML
