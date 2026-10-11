# Kiến Trúc Hệ Thống & Tối Ưu Big Data

Tài liệu này mô tả kiến trúc tổng thể, mô hình dữ liệu Medallion, và các kỹ thuật tối ưu hóa phần cứng / Apache Spark để xử lý 82 triệu dòng dữ liệu trên máy tính cá nhân chạy Windows với 16 GB RAM.

---

## 1. Sơ Đồ Kiến Trúc Medallion

Hệ thống tuân thủ mô hình kiến trúc Medallion 4 tầng tiêu chuẩn trong kỹ nghệ dữ liệu lớn:

```mermaid
flowchart TD
    subgraph Raw["1. Raw Data Layer"]
        R1["itineraries.csv (31 GB, 82.138.753 dòng)"]
        R2["7 Nguồn Dữ Liệu Ngoài (API, CSV, XLS)"]
    end

    subgraph Bronze["2. Bronze Layer (Data Lake)"]
        B1["itineraries/ (Parquet zstd, 4.5 GB)"]
        B2["external/clean/ (7 bảng chuẩn hóa)"]
    end

    subgraph Silver["3. Silver Layer (Cleaned & Enriched)"]
        S1["itineraries/ (Parquet zstd, 57 cột)\nPartition by flight_month\n82.137.463 dòng sạch"]
    end

    subgraph Gold["4. Gold Layer (Aggregations & Features)"]
        G1["21 Bảng Thống Kê & Phân Tích Chiều Sâu"]
        G2["Ma Trận Tương Quan (Pearson / Spearman / Eta)"]
        G3["feature_selection.json & best_model.json"]
    end

    subgraph Serving["5. Serving & Application Layer"]
        M1["Spark MLlib Models (GBT, RF, LR, DT)"]
        M2["Dự Báo 2 Tầng (SARIMAX + Vi Mô Expedia)"]
        M3["Bảng Kết Quả CSV (exports/ ~25 files)"]
        D1["Interactive HTML Dashboard (index.html)"]
        F1["Báo Cáo Biểu Đồ (report/figures/)"]
    end

    R1 -->|Schema khai báo trước, Permissive mode| B1
    R2 -->|ETL Pandas / Regex / API Call| B2
    B1 & B2 -->|Filter, Feature Engineering, Broadcast Join| S1
    S1 -->|Spark SQL Aggregations & MLlib Correlation| G1 & G2
    G2 --> G3
    S1 & G3 -->|Train 4-8, Test 9-10 (5% Sample)| M1
    S1 & B2 & G1 -->|Backtest 2023, Elasticity, Forecast 2027| M2
    G1 & M1 & M2 --> M3
    M3 --> D1 & F1
```

---

## 2. Đặc Tả Từng Tầng Lưu Trữ

### Tầng 1: Raw Data
- **File gốc:** `itineraries.csv` từ Expedia (dataset Kaggle `dilwong/flightprices`), kích thước ~31 GB, chứa 82.138.753 dòng và 27 cột. Dữ liệu ghi nhận các lượt tra cứu vé máy bay từ 16/04/2022 đến 05/10/2022 với ngày bay từ 17/04/2022 đến 19/11/2022.
- **Dữ liệu bổ trợ:** 7 nguồn vĩ mô và vi mô tải về qua HTTP API và các kho dữ liệu công cộng (OpenFlights, holidays, Open-Meteo, EIA, BLS, DOT Table 1a, BTS).

### Tầng 2: Bronze Layer (Bảo toàn nguyên trạng)
- **Cơ chế nạp:** Nạp trực tiếp vào Parquet thông qua PySpark bằng cách **khai báo schema tường minh** (`StructType` gồm 27 trường). Tuyệt đối không bật `inferSchema=True` vì việc quét 31 GB CSV chỉ để đoán kiểu dữ liệu sẽ gây tràn đĩa đệm và tiêu tốn hàng chục phút.
- **Nén:** Sử dụng thuật toán nén **zstd (Zstandard)** cấp độ tối ưu, nén dữ liệu từ 31 GB thô xuống còn **4,5 GB** trên đĩa.
- **Nguyên tắc Bronze:** Giữ nguyên 100% dòng và cột (kể cả dòng có giá trị lỗi, âm, hoặc thiếu), đảm bảo khả năng tái tạo (reproducibility) khi thay đổi logic nghiệp vụ.

### Tầng 3: Silver Layer (Làm sạch & Tích hợp)
- **Lọc dữ liệu rác:** Bỏ các bản ghi có giá `totalFare <= 0`, ngày tra cứu hoặc ngày bay bị `NULL`, ngày bay trước ngày tra cứu (`days_before < 0`).
- **Khử trùng lặp hiệu năng cao:** Bỏ trùng lặp dựa trên khóa logic `["legId", "searchDate", "fareBasisCode", "totalFare"]`. *Kỹ thuật tối ưu:* Thực hiện phép chiếu (`select`) các cột cần thiết trước khi gọi `dropDuplicates()` để Spark chỉ phải shuffle các cột kích thước nhỏ, loại bỏ các chuỗi hành trình dài (`segments...`) giúp tiết kiệm hơn 80% I/O đĩa tạm.
- **Tạo đặc trưng nghiệp vụ:** Trích xuất giờ bay, thứ trong tuần, tháng bay, số chặng dừng, hạng ghế (`cabin_level`), nhóm thời điểm đặt vé (`booking_window`: 1–7 ngày, 8–30 ngày, 31–60 ngày).
- **Tích hợp nguồn ngoài bằng Broadcast Join:** 7 bảng nguồn ngoài đều có kích thước nhỏ (từ vài chục KB đến vài MB). Sử dụng `F.broadcast()` để phát tán các bảng này tới toàn bộ executor/task, loại bỏ hoàn toàn giai đoạn Shuffle Join đắt đỏ trên 82 triệu dòng.
- **Phân vùng (Partitioning):** Phân vùng theo tháng bay (`partitionBy("flight_month")`). Nhờ đó, các tác vụ phân tích hay chia tập dữ liệu huấn luyện (T4–T8) và kiểm tra (T9–T10) chỉ đọc đúng thư mục phân vùng tương ứng (Partition Pruning), tiết kiệm triệt để thời gian đọc đĩa.
- **Quy mô sau Silver:** **82.137.463 dòng sạch** (loại 1.290 dòng lỗi/trùng), 57 cột, dung lượng ~5,8 GB Parquet.

### Tầng 4: Gold Layer (Tổng hợp & Lựa chọn chỉ số)
- **21 Bảng Gold:** Tính toán các chỉ số thống kê (`count`, `avg`, `median` xấp xỉ qua `percentile_approx`, phân vị 25%, 75%) theo từng lát cắt: số ngày đặt trước, hãng bay, tuyến bay, hạng ghế, số chặng dừng, thời tiết, ngày lễ, v.v.
- **Ma trận tương quan toàn diện:**
  - Pearson tính toán trực tiếp trên toàn bộ 82M dòng bằng `pyspark.ml.stat.Correlation`.
  - Spearman tính trên mẫu 3% (do thuật toán ranking phân tán đòi hỏi chi phí tính toán lớn).
  - Tỷ số tương quan phương sai $\eta$ (Eta) cho các biến định tính (tuyến bay, hãng hàng không, sân bay).
- **Bộ chỉ số được chọn (`feature_selection.json`):** Lưu trữ các đặc trưng vượt qua ngưỡng tương quan, loại bỏ các biến trùng lặp thông tin ($|r| > 0,9$) và các biến rò rỉ đáp án như `fareBasisCode`.

---

## 3. Chiến Lược Tối Ưu Hóa PySpark Trên Windows 16 GB RAM

Chạy một pipeline Spark trên 82 triệu bản ghi trong môi trường Windows local gặp phải 3 nút thắt chính: **Tràn RAM (OOM)**, **Đầy đĩa cứng (Disk Spill)**, và **Tương thích Hadoop Winutils**. Mã nguồn tại [common.py](file:///f:/PKA/Code/BD/flight-price-analysis/common.py) triển khai các cơ chế tối ưu đặc thù:

### 1. Phân bổ bộ nhớ Driver (`driver_memory = "9g"`)
- Với RAM vật lý 16 GB, cấp phát 9 GB cho Spark Driver process, chừa lại 7 GB cho hệ điều hành Windows, background services, và bộ đệm file hệ thống (OS page cache).

### 2. Định tuyến đĩa đệm thông minh (`_pick_best_data_dir()`)
- Khi Spark shuffle hoặc spill dữ liệu ra đĩa, thư mục đệm có thể phình to từ 10 đến 30 GB.
- Hàm `_pick_best_data_dir()` tự động dò quét các phân vùng ổ đĩa (E:, F:, D:, C:), kiểm tra dung lượng trống khả dụng và tự động chọn phân vùng có dung lượng trống lớn nhất (tối thiểu 5 GB).
- Cấu hình `spark.local.dir` trỏ trực tiếp vào thư mục đệm trên phân vùng được chọn, tránh làm đầy ổ C: của hệ thống.

### 3. Điều chỉnh số phân vùng xáo trộn (`spark.sql.shuffle.partitions = "96"`)
- Mặc định của Spark là 200 phân vùng. Với 12 luồng CPU cục bộ, số phân vùng được đặt thành **96** (bội số của 12 luồng: $12 \times 8$).
- Điều này chia nhỏ khối lượng công việc của từng task, vừa vặn với kích thước heap 9 GB, hạn chế tối đa tình trạng GC pause (Garbage Collection) kéo dài.

### 4. Tăng tốc Arrow và Giảm thiểu log rác
- `spark.sql.execution.arrow.pyspark.enabled = true`: Tối ưu hóa việc chuyển đổi dữ liệu giữa JVM của Spark và Pandas dataframe qua bộ nhớ chia sẻ Apache Arrow.
- `sparkContext.setLogLevel("ERROR")` và `spark.ui.showConsoleProgress = false`: Loại bỏ hàng triệu dòng log tiến trình ghi vào stdout của Windows PowerShell, giảm tắc nghẽn I/O console.

### 5. Checkpoint cho các mô hình dạng cây
- Khi huấn luyện Random Forest hoặc Gradient-Boosted Trees trong Spark MLlib, đồ thị tính toán (lineage graph) sẽ phình to theo từng độ sâu của cây.
- Cấu hình `checkpointInterval=10` kết hợp `sparkContext.setCheckpointDir` cắt tỉa cây tính toán, giải phóng bộ nhớ heap định kỳ.

### 6. Cấu hình Windows Native Hadoop (`winutils.exe` & `hadoop.dll`)
- Spark trên Windows yêu cầu `winutils.exe` để quản lý quyền tập tin POSIX ảo.
- [common.py](file:///f:/PKA/Code/BD/flight-price-analysis/common.py#L9-L18) tự động phát hiện `HADOOP_HOME`, nạp `hadoop.dll` vào biến môi trường `PATH`, khắc phục triệt để lỗi `java.io.IOException: Could not locate executable null\bin\winutils.exe`.

---

## 4. Thời Gian Thực Thi Trên Phần Cứng Tham Chiếu

*(Cấu hình kiểm thử: CPU 12 luồng, 16 GB RAM DDR4, SSD NVMe)*

| Module | Nhiệm vụ | Thời gian chạy |
|---|---|---|
| `01_bronze_ingest` | Đọc 31 GB CSV, nén zstd, ghi 4,5 GB Parquet | ~3 phút |
| `02_external_sources` | Tải và chuẩn hóa 7 nguồn dữ liệu ngoài | ~2 phút |
| `03_silver_etl` | Làm sạch, feature engineering, broadcast join | ~11 phút |
| `04_gold_analysis_correlation` | 21 bảng Gold + Pearson/Spearman/$\eta$ trên 82M dòng | ~40 phút |
| `05_ml_models` | Pipeline 5 mô hình MLlib, ablation study trên 5% mẫu | ~19 phút |
| `06_forecast_2023_2027` | SARIMAX vĩ mô, backtest DOT, dự báo 2027 | ~2 phút |
| `build_dashboard` | Trích xuất export CSV $\to$ HTML độc lập | ~3 giây |
| **Tổng cộng toàn bộ pipeline** | | **~77 phút** |
