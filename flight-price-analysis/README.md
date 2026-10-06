# Phân tích giá vé máy bay Mỹ 2022 → 2027 (PySpark)

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `common.py` | Đường dẫn dữ liệu và hàm `get_spark()` (Spark local trên Windows, file tạm ở ổ E) |
| `viz.py` | Style biểu đồ dùng chung |
| `notebooks/01_bronze_ingest` | CSV 31 GB → Parquet (Bronze), 82.138.753 dòng |
| `notebooks/02_external_sources` | 7 nguồn ngoài: sân bay, ngày lễ, thời tiết, nhiên liệu EIA, CPI BLS, giá vé DOT, BTS |
| `notebooks/03_silver_etl` | Làm sạch, tạo đặc trưng, tích hợp → Silver (57 cột, phân vùng theo tháng bay) |
| `notebooks/04_gold_analysis_correlation` | 21 bảng Gold + Pearson/Spearman/η trên toàn bộ dữ liệu, chọn chỉ số |
| `notebooks/05_ml_models` | Học tháng 4–8, kiểm tra tháng 9–10/2022; so sánh 5 mô hình; thí nghiệm tích hợp |
| `notebooks/06_forecast_2023_2027` | Backtest 2023 vs DOT, hiệu chỉnh, kiểm tra Q1/2024, dự báo đến 12/2027 |
| `dashboard/index.html` | Dashboard HTML (mở trực tiếp bằng trình duyệt, cần internet để tải Chart.js và font) |
| `report/figures/` | Biểu đồ PNG dùng cho báo cáo và slide |

Mỗi notebook có hai dạng: `.py` (mã nguồn, định dạng `# %%`) và `.ipynb` (đã chạy, có sẵn kết quả).

## Dữ liệu

- Gốc: `D:/TH-PTDLL/data/flightprices/itineraries.csv`
- Bronze / Silver / Gold / mô hình: `E:/flight-data/`
- Bảng kết quả CSV để nộp: `E:/flight-data/exports/`

## Chạy lại

```bash
cd flight-price-analysis
set PYTHONUTF8=1
python build_notebooks.py 01      # hoặc 02, 03 … 06; chuyển .py → .ipynb và chạy
python dashboard/build_dashboard.py
```

Yêu cầu: Python 3.12, `pyspark==4.0.1`, `statsmodels`, `holidays`, `pandas`, `matplotlib`, `xlrd`, `nbclient`;
Java 17; `winutils.exe` + `hadoop.dll` ở `E:/hadoop/bin`.

Thời gian chạy trên máy 16 GB RAM / 12 luồng: 01 ≈ 3 phút, 03 ≈ 11 phút, 04 ≈ 40 phút, 05 ≈ 19 phút, 06 ≈ 2 phút.
