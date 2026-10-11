# Hướng Dẫn Vận Hành & Thực Thi (Runbook)

Tài liệu này cung cấp hướng dẫn từng bước để thiết lập môi trường, thực thi toàn bộ pipeline phân tích dữ liệu, xuất báo cáo và khởi chạy dashboard tương tác.

---

## 1. Yêu Cầu Môi Trường (Prerequisites)

### 1.1. Phần mềm cốt lõi
- **Hệ điều hành:** Windows 10/11 (64-bit).
- **Python:** Phiên bản `3.12+` (khuyến nghị kích hoạt `PYTHONUTF8=1` để tránh lỗi mã hóa ký tự trên console Windows).
- **Java Development Kit:** **JDK 25 LTS** (hoặc JDK 17). Đã cài đặt tại `C:\Program Files\Java\jdk-25.0.2`.
- **Hadoop Winutils trên Windows:**
  - Tệp `winutils.exe` và `hadoop.dll` đặt tại `F:/PKA/Code/DLL/hadoop/bin`.
  - [common.py](file:///f:/PKA/Code/BD/flight-price-analysis/common.py#L9-L26) tự động kiểm tra và đưa vào biến môi trường `PATH`.

### 1.2. Thư viện Python cần thiết
Cài đặt các gói phụ thuộc qua pip:

```bash
pip install pyspark pandas numpy matplotlib statsmodels holidays requests xlrd openpyxl nbformat nbclient
```

---

## 2. Cấu Hình Đường Dẫn Dữ Liệu Thực Tế (Data Path Configuration)

Hệ thống được thiết kế tự động phát hiện đường dẫn thực tế trên máy tại [common.py](file:///f:/PKA/Code/BD/flight-price-analysis/common.py):

1. **Tệp CSV thô (`CSV_PATH`):**
   - Đang lưu trữ tại: `C:\Users\phapd\.cache\kagglehub\datasets\dilwong\flightprices\versions\1\itineraries.csv` (31,09 GB).
2. **Thư mục lưu trữ chính (`DATA_DIR`):**
   - Đặt tại `F:/flight-data` (phân vùng ổ `F:` có 95 GB dung lượng trống).
   - Có thể chỉ định thủ công bằng biến môi trường `FLIGHT_DATA_DIR`.
   - Cấu trúc thư mục con tự động khởi tạo:
     - `bronze/itineraries`: Parquet dữ liệu thô.
     - `silver/itineraries`: Parquet dữ liệu đã làm sạch & tích hợp (phân vùng theo `flight_month`).
     - `gold/`: 21 bảng tổng hợp Parquet & file cấu hình JSON.
     - `external/clean`: 7 bảng nguồn ngoài dạng CSV nhỏ.
     - `exports/`: Tất cả các bảng CSV tổng hợp phục vụ nộp bài & dashboard.
     - `spark-tmp/`: Đĩa đệm shuffle và checkpoint cho Spark.

---

## 3. Quy Trình Thực Thi Từng Bước (Execution Workflow)

### Bước 0: Tạo mẫu dữ liệu nhỏ để kiểm tra nhanh (Tùy chọn)
Nếu muốn mở file trên Excel hoặc kiểm tra nhanh mà không cần đọc 31 GB:

```bash
cd f:/PKA/Code/BD/flight-price-analysis
python make_samples.py
```
*Kết quả:* Tạo 2 file tại `samples/`:
- `itineraries_first_1000.csv` (1.000 dòng đầu).
- `itineraries_sample_100k.csv` (100.000 dòng phân bổ đều toàn file).

---

### Bước 1 $\to$ 6: Chạy Pipeline Phân Tích (PySpark & ML)
Sử dụng công cụ `build_notebooks.py` để tự động chuyển đổi mã nguồn `.py` thành `.ipynb` (Jupyter Notebook), thực thi và lưu sẵn kết quả hiển thị:

```bash
cd f:/PKA/Code/BD/flight-price-analysis
set PYTHONUTF8=1

# 1. Nạp CSV thô 31 GB vào tầng Bronze Parquet zstd (khoảng 3 phút)
python build_notebooks.py 01

# 2. Tải và chuẩn hóa 7 nguồn dữ liệu ngoài (khoảng 2 phút)
python build_notebooks.py 02

# 3. Làm sạch, feature engineering và tích hợp tầng Silver (khoảng 11 phút)
python build_notebooks.py 03

# 4. Tính toán 21 bảng Gold và ma trận tương quan trên toàn bộ 82M dòng (khoảng 40 phút)
python build_notebooks.py 04

# 5. Huấn luyện 5 mô hình MLlib và ablation test trên 5% mẫu (khoảng 19 phút)
python build_notebooks.py 05

# 6. Dự báo SARIMAX, backtest DOT 2023, hiệu chỉnh và dự báo 2027 (khoảng 2 phút)
python build_notebooks.py 06
```

> **Mẹo:** Nếu chỉ muốn tạo file `.ipynb` để tự mở trên VS Code / Jupyter Lab mà không muốn chạy tự động trên dòng lệnh, thêm cờ `--no-run`:
> ```bash
> python build_notebooks.py 01 02 03 04 05 06 --no-run
> ```

---

### Bước 7: Xây dựng & Mở Dashboard Tương Tác

Sau khi các bảng kết quả CSV đã được xuất ra thư mục `exports/`:

```bash
cd f:/PKA/Code/BD/flight-price-analysis
python dashboard/build_dashboard.py
```

*Kết quả tạo ra:*
- [dashboard/index.html](file:///f:/PKA/Code/BD/flight-price-analysis/dashboard/index.html): Trang Dashboard độc lập (Self-contained HTML).
- **Cách xem:** Click đúp chuột vào [index.html](file:///f:/PKA/Code/BD/flight-price-analysis/dashboard/index.html) để mở trực tiếp trên Google Chrome, Microsoft Edge hoặc Firefox. Toàn bộ dữ liệu phân tích đã được nhúng sẵn dưới dạng JSON, chỉ cần kết nối internet để tải thư viện Chart.js và Google Fonts từ CDN.

---

## 4. Danh Mục Sản Phẩm Bàn Giao (Deliverables Checklist)

| Nhóm sản phẩm | Vị trí | Mô tả |
|---|---|---|
| **Mã nguồn Pipeline** | `flight-price-analysis/notebooks/*.py` | 6 kịch bản xử lý Spark từ Ingestion đến Forecasting |
| **Sổ tay có sẵn kết quả** | `flight-price-analysis/notebooks/*.ipynb` | 6 Jupyter Notebook đã chạy hoàn chỉnh kèm bảng & biểu đồ |
| **Dashboard tương tác** | `flight-price-analysis/dashboard/index.html` | Dashboard HTML5/Chart.js 7 phân hệ phân tích |
| **Biểu đồ báo cáo (PNG)**| `flight-price-analysis/report/figures/` | 11 biểu đồ độ phân giải cao phục vụ chèn Word/Slide |
| **Bảng số liệu Export** | `flight-data/exports/*.csv` | ~25 bảng CSV tổng hợp phục vụ tra cứu và kiểm toán |

---

## 5. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 1. Lỗi `java.lang.OutOfMemoryError: Java heap space`
- **Nguyên nhân:** Driver memory bị cấp phát quá mức hoặc garbage collection không kịp thu hồi.
- **Khắc phục:** Mở [common.py](file:///f:/PKA/Code/BD/flight-price-analysis/common.py#L61) giảm `driver_memory="9g"` xuống `7g` hoặc `8g`. Đảm bảo đóng bớt các ứng dụng chiếm nhiều RAM (trình duyệt, máy ảo) trước khi chạy module 04 và 05.

### 2. Lỗi `No space left on device` trong lúc shuffle
- **Nguyên nhân:** Phân vùng ổ đĩa chứa thư mục tạm bị đầy do dữ liệu spill.
- **Khắc phục:** Đặt biến môi trường `FLIGHT_DATA_DIR` trỏ sang phân vùng ổ đĩa còn nhiều dung lượng trống (như ổ `F:` còn 95 GB):
  ```bash
  set FLIGHT_DATA_DIR=F:/flight-data
  ```

### 3. Lỗi `Could not locate executable null\bin\winutils.exe`
- **Khắc phục:** Đảm bảo đường dẫn `HADOOP_HOME` chứa thư mục `bin\winutils.exe` hợp lệ (đã được cấu hình tự động tại `common.py`).

### 4. Lỗi hiển thị tiếng Việt trên Terminal PowerShell
- **Khắc phục:** Trước khi chạy script Python, thực thi lệnh PowerShell:
  ```powershell
  $OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
  $env:PYTHONUTF8 = "1"
  ```
