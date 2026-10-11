## 1. Dữ Liệu Gốc & Đường Dẫn Đã Thiết Lập

- **Tệp CSV thô 31 GB:** `F:/PKA/Code/BD/itineraries.csv` (31,09 GB - 31.091.834.438 bytes).
- **Thư mục lưu trữ dữ liệu:** `F:/flight-data` (trên phân vùng ổ `F:` còn 64 GB trống).
- **Tầng Bronze & Silver:** Sẽ được tự động nạp và ghi ra Parquet zstd tại `F:/flight-data/bronze` và `F:/flight-data/silver` (82.137.463 dòng).
- **Tầng Gold:** 21 bảng tổng hợp và ma trận tương quan Pearson trên toàn bộ 82M dòng lưu tại `F:/flight-data/gold`.

---

## 2. Cấu Hình Phần Cứng & Phân Bổ Bộ Nhớ Tối Đa (32 GB RAM)

Tệp [common_full.py](file:///f:/PKA/Code/BD/flight-price-full-training/common_full.py) được cấu hình mức bộ nhớ tối đa:

| Thành phần | Cấu hình | Mục đích kỹ thuật |
|---|---|---|
| **CPU Threads** | `local[20]` | Tận dụng toàn bộ 20 luồng (6 P-cores + 8 E-cores của i5-14500). |
| **Spark Driver Heap** | `spark.driver.memory = 24g` | Cấp phát tối đa 24 GB cho JVM Heap (chứa dữ liệu và ma trận đặc trưng lớn). |
| **Hệ thống còn lại** | `8 GB` RAM trống | Dành cho Windows 11, background processes và OS page cache. |
| **Shuffle Partitions** | `spark.sql.shuffle.partitions = 160` | $160 = 20 \text{ luồng} \times 8$: chia nhỏ kích thước mỗi task để tránh tràn heap. |
| **Bộ dọn rác (GC)** | `G1GC` (InitiatingHeapOccupancyPercent=35) | Kích hoạt dọn rác nền sớm, triệt tiêu hiện tượng Stop-The-World trên heap 24 GB. |

---

## 3. Hướng Dẫn Thực Thi

### Cách 1: Chạy trọn gói qua Windows Batch (Khuyến nghị)
Click đúp chuột vào file [run_training.bat](file:///f:/PKA/Code/BD/flight-price-full-training/run_training.bat):
- Chọn `1`: Chạy tự động nối tiếp **Bronze $\to$ Silver $\to$ Gold $\to$ Machine Learning** trên 100% dữ liệu.

### Cách 2: Chạy trực tiếp từ PowerShell / Terminal

```bash
cd f:/PKA/Code/BD/flight-price-full-training
set PYTHONUTF8=1

# Chạy quy trình từ Silver đến Gold đến Machine Learning với 24 GB RAM
python pipeline_silver_to_gold_ml.py
```

---

## 4. Thời Gian Chạy Tham Chiếu & Tài Nguyên Dự Kiến

*(Đo lường trên Intel Core i5-14500, RAM 32 GB DDR5, SSD NVMe)*

| Mức dữ liệu | Số dòng Train (T04–T08) | Số dòng Test (T09–T10) | Thời gian chạy ước tính | Mức tiêu thụ RAM đỉnh |
|---|---|---|---|---|
| **20% Dữ liệu** | ~10,7 triệu dòng | ~5,7 triệu dòng | **~10 – 15 phút** | ~14 – 16 GB |
| **50% Dữ liệu** | ~26,9 triệu dòng | ~14,2 triệu dòng | **~28 – 38 phút** | ~18 – 22 GB |
| **100% Dữ liệu** | **~53,8 triệu dòng** | **~28,3 triệu dòng** | **~65 – 90 phút** | ~24 – 26 GB |

---

## 5. Giám Sát Tiến Trình Thời Gian Thực

Trong lúc mã nguồn đang chạy, bạn có thể mở trình duyệt web và truy cập:
👉 **`http://localhost:4040`**

Giao diện **Spark Web UI** sẽ hiển thị:
- **Jobs / Stages:** Số lượng task đang chạy song song trên 20 luồng CPU.
- **Executors:** Biểu đồ sử dụng bộ nhớ JVM Heap và thời gian dọn rác (GC Time).
- **Storage:** Kích thước các phân vùng Parquet đang được đọc vào RAM.

---

## 6. Kết Quả Đầu Ra

Sau khi hoàn tất, kết quả được xuất ra tại thư mục:
- Bảng đối chiếu mô hình: `F:/flight-data/exports_full_32gb/model_comparison_full.csv`
- Mức độ quan trọng đặc trưng: `F:/flight-data/exports_full_32gb/feature_importance_full.csv`
- Mô hình Spark MLlib tốt nhất đã lưu: `F:/flight-data/models_full_32gb/best_model_full`
