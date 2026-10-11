# Hệ Thống Phân Tích & Dự Báo Giá Vé Máy Bay Mỹ (2022 → 2027)

Dự án xử lý dữ liệu lớn (Big Data) ứng dụng kiến trúc **Medallion (Bronze – Silver – Gold)** trên nền tảng **Apache Spark (PySpark 4.0.1)** để phân tích toàn bộ **82.138.753 dòng dữ liệu vé máy bay** (31 GB CSV thô), tích hợp 7 nguồn dữ liệu bên ngoài (vĩ mô, thời tiết, nhiên liệu, ngày lễ), huấn luyện mô hình Machine Learning và xây dựng hệ thống dự báo giá vé đa tầng đến năm 2027.

---

## 1. Mục Tiêu & Điểm Nhấn Dự Án

- **Xử lý toàn bộ 82,1 triệu dòng trên máy tính cá nhân (16 GB RAM):** Sử dụng PySpark local tối ưu bộ nhớ, định dạng lưu trữ cột nén Parquet zstd (giảm từ 31 GB CSV xuống 4,5 GB Parquet), cơ chế broadcast join và phân vùng `flight_month`.
- **Tích hợp dữ liệu đa nguồn (Multi-source Enrichment):** Ghép nối dữ liệu chuyến bay với 7 nguồn ngoài gồm OpenFlights (sân bay/tọa độ), US Holidays (ngày lễ liên bang), Open-Meteo (thời tiết lịch sử theo ngày/sân bay), EIA (giá dầu phản lực Jet Fuel), BLS (CPI vé máy bay & CPI chung từ 2005 đến nay), DOT Consumer Airfare Report (giá thực tế theo tuyến 1993–2024), BTS (giá vé theo sân bay).
- **Phân tích hành vi định giá (Booking Curve & Urgency Premium):** Khảo sát đường cong đặt vé trước 1–60 ngày trên toàn bộ dữ liệu. Chứng minh vé đặt sát ngày bay (1–7 ngày) có giá trung vị cao hơn 19% so với đặt trước 31–60 ngày, cá biệt các chặng ngắn hoặc hãng bay giá rẻ (LCC) đội giá từ 40% đến hơn 100%.
- **Lựa chọn đặc trưng trên Big Data:** Tính toán hệ số tương quan Pearson trên toàn bộ 82M dòng (Spark MLlib), Spearman trên mẫu 3%, và tỷ số tương quan $\eta$ (Eta) cho các biến định tính. Loại bỏ biến đa cộng tuyến và biến rò rỉ thông tin trước khi huấn luyện mô hình.
- **Machine Learning Out-of-Time:** Huấn luyện trên các chuyến bay tháng 04–08/2022, kiểm tra trên tháng 09–10/2022 (mẫu 5% ngẫu nhiên). So sánh 5 thuật toán (Baseline, Linear Regression, Decision Tree, Random Forest, GBT Regressor). Thử nghiệm chứng minh việc bổ sung dữ liệu ngoài giúp giảm sai số MAE/MAPE rõ rệt.
- **Dự báo giá vé 2 tầng (Hybrid Micro-Macro Forecasting 2023 → 2027):** 
  - *Tầng vi mô:* Mức giá cơ sở từng tuyến và quy luật booking curve theo ngày từ 82M dòng Expedia 2022.
  - *Tầng vĩ mô:* Xu hướng giá, lạm phát và chu kỳ qua mô hình SARIMAX chuỗi thời gian kết hợp hệ số co giãn giá vé DOT–CPI ($r \approx 0,82$).
  - *Backtest 2023:* Dự báo 2023 từ dữ liệu 2022 rồi đối chiếu trực tiếp với dữ liệu thực tế DOT 2023 ($MAPE \approx 7,8\% - 9,2\%$). Áp dụng hệ số hiệu chỉnh độ lệch (bias calibration + route shrinkage).
  - *Kiểm chứng Q1/2024:* Đo lường trên thực tế Q1/2024 đạt sai số $MAPE \approx 8,1\%$.
  - *Dự báo tương lai:* Xuất chuỗi CPI dự báo và đường giá vé từng ngày đặt trước cho 234 tuyến bay đến 12/2027.
- **Sản phẩm đầu ra trực quan:** Dashboard HTML tương tác độc lập ([index.html](file:///f:/PKA/Code/BD/flight-price-analysis/dashboard/index.html)), tự chứa toàn bộ dữ liệu phân tích, không phụ thuộc server backend.

---

## 2. Hệ Thống Tài Liệu Chi Tiết

Tài liệu dự án được cấu trúc theo các khía cạnh kỹ thuật chuyên sâu:

| Tài liệu | Nội dung chính |
|---|---|
| [ARCHITECTURE.md](file:///f:/PKA/Code/BD/docs/ARCHITECTURE.md) | Kiến trúc Medallion (Bronze/Silver/Gold), thiết kế hệ thống, cơ chế tối ưu PySpark trên Windows (16 GB RAM, local storage). |
| [DATA_PIPELINE.md](file:///f:/PKA/Code/BD/docs/DATA_PIPELINE.md) | Luồng Ingestion, chi tiết 7 nguồn dữ liệu ngoài, quy trình ETL Silver, sơ đồ 57 trường dữ liệu và tỷ lệ join match. |
| [EXPLORATORY_ANALYSIS.md](file:///f:/PKA/Code/BD/docs/EXPLORATORY_ANALYSIS.md) | Phân tích 82M dòng: booking curve, mức đội giá khi mua gấp, ma trận tương quan Pearson/Spearman, tỷ số $\eta$, tiêu chí chọn đặc trưng. |
| [MACHINE_LEARNING.md](file:///f:/PKA/Code/BD/docs/MACHINE_LEARNING.md) | Huấn luyện MLlib out-of-time (T4–T8 $\to$ T9–T10), kiểm tra phân phối giá trị ngoài khoảng, so sánh 5 mô hình, ablation study dữ liệu ngoài. |
| [FORECASTING_2023_2027.md](file:///f:/PKA/Code/BD/docs/FORECASTING_2023_2027.md) | Phương pháp dự báo lai 2 tầng: SARIMAX vĩ mô + vi mô Expedia, backtest 2023 vs DOT, hiệu chỉnh co giãn, kiểm chứng Q1/2024, dự báo 2026–2027. |
| [RUNBOOK.md](file:///f:/PKA/Code/BD/docs/RUNBOOK.md) | Hướng dẫn cài đặt môi trường (Java 17, Hadoop Winutils, PySpark), quy trình thực thi pipeline và hướng dẫn mở dashboard. |

---

## 3. Cấu Trúc Thư Mục Dự Án

```text
f:/PKA/Code/BD/
├── README.md                                  # Trang tổng quan dự án (file này)
├── .gitignore                                 # Loại trừ thư mục data/
├── docs/                                      # Bộ tài liệu kiến thức chuyên sâu
│   ├── ARCHITECTURE.md
│   ├── DATA_PIPELINE.md
│   ├── EXPLORATORY_ANALYSIS.md
│   ├── MACHINE_LEARNING.md
│   ├── FORECASTING_2023_2027.md
│   └── RUNBOOK.md
└── flight-price-analysis/
    ├── README.md                              # Tóm tắt nhanh pipeline và thời gian chạy
    ├── common.py                              # Cấu hình SparkSession, biến môi trường, đường dẫn đĩa
    ├── viz.py                                 # Định chuẩn kiểu biểu đồ Matplotlib đồng nhất
    ├── build_notebooks.py                     # Script chuyển đổi .py sang .ipynb và chạy tự động
    ├── make_samples.py                        # Trích xuất mẫu nhỏ 1.000 dòng và 100k dòng từ CSV 31 GB
    ├── notebooks/                             # 6 module phân tích cốt lõi (song song file .py và .ipynb)
    │   ├── 01_bronze_ingest.py (.ipynb)       # CSV 31 GB -> Bronze Parquet zstd (82.138.753 dòng)
    │   ├── 02_external_sources.py (.ipynb)    # Thu thập & chuẩn hóa 7 nguồn dữ liệu ngoài
    │   ├── 03_silver_etl.py (.ipynb)          # Làm sạch, tạo đặc trưng, broadcast join -> Silver Parquet
    │   ├── 04_gold_analysis_correlation.py    # 21 bảng Gold + Ma trận Pearson/Spearman/Eta trên 82M dòng
    │   ├── 05_ml_models.py (.ipynb)           # Spark MLlib: Train T4-T8, Test T9-T10, so sánh 5 mô hình
    │   └── 06_forecast_2023_2027.py (.ipynb)  # Backtest 2023, hiệu chỉnh, kiểm tra 2024, dự báo 2027
    ├── dashboard/                             # Ứng dụng Dashboard HTML tương tác
    │   ├── build_dashboard.py                 # Nhúng dữ liệu từ exports/ vào template HTML
    │   ├── template.html                      # Giao diện HTML5/CSS3/Chart.js
    │   ├── artifact.html                      # Phiên bản HTML nhúng
    │   └── index.html                         # Dashboard độc lập mở trực tiếp trên trình duyệt
    └── report/
        └── figures/                           # 11 biểu đồ độ phân giải cao phục vụ báo cáo
└── flight-price-full-training/                # Module huấn luyện mở rộng cho máy 32 GB RAM (i5-14500)
    ├── README.md                              # Hướng dẫn chi tiết huấn luyện 100% dữ liệu (82M dòng)
    ├── common_full.py                         # Cấu hình Spark 22GB Heap, 20 luồng, G1GC tuning
    ├── train_full.py                          # Pipeline huấn luyện ML quy mô lớn (hỗ trợ fraction 0.2 - 1.0)
    └── run_training.bat                       # Script menu thực thi 1-click trên Windows
```

---

## 4. Các Chỉ Số Cốt Lõi (Key Metrics Summary)

- **Quy mô tập dữ liệu sau làm sạch (Silver):** 82.137.463 bản ghi giá vé, 5.999.739 hành trình (`legId`), 234 tuyến bay trực tiếp/gián tiếp kết nối 16 sân bay lớn nhất nước Mỹ, 14 hãng hàng không.
- **Mức giá tổng quan 2022:** Giá trung bình **$340.39**, giá trung vị **$305.20**.
- **Quy luật đặt chỗ (Booking Window):**
  - Đặt sớm (31–60 ngày): Giá trung vị **$288.10**.
  - Đặt gấp (8–30 ngày): Giá trung vị **$305.61** (+6,1%).
  - Đặt rất gấp (1–7 ngày): Giá trung vị **$342.60** (+18,9%).
- **Mô hình Machine Learning (Spark MLlib):**
  - Thuật toán tối ưu: **Gradient-Boosted Trees (GBT)** / **Random Forest**.
  - Sai số trên tập kiểm tra tháng 09–10/2022 (chưa từng thấy trong lúc học): $MAE \approx 45,6\text{ USD}$, $MAPE \approx 13,8\%$, $R^2 \approx 0,81$.
- **Mô hình Dự báo 2023–2027:**
  - Sai số kiểm chứng Backtest 2023 so với giá thực tế DOT: $MAPE \approx 7,8\% - 9,2\%$.
  - Sai số kiểm chứng ngoại suy Q1/2024: $MAPE \approx 8,1\%$.
  - Xu hướng CPI vé máy bay: Tăng từ mức 284,8 (2022) đạt đỉnh chu kỳ và dự báo ổn định quanh mức tăng trưởng lạm phát dài hạn (+2,5% đến +3,5%/năm) đến cuối 2027.
