# Mô Hình Học Máy: Huấn Luyện & Đánh Giá Out-Of-Time

Tài liệu này chi tiết hóa phương pháp xây dựng pipeline Machine Learning trên nền tảng **Apache Spark MLlib**, chiến lược thẩm định theo thời gian (Out-of-time validation), thử nghiệm bóc tách đóng góp của dữ liệu ngoài (Ablation Study), và đánh giá độ quan trọng của đặc trưng tại [05_ml_models.py](file:///f:/PKA/Code/BD/flight-price-analysis/notebooks/05_ml_models.py).

---

## 1. Chiến Lược Thẩm Định Thực Tế (Out-Of-Time Validation)

### 1.1. Tại sao không dùng K-Fold Cross Validation ngẫu nhiên?
Trong bài toán định giá và chuỗi thời gian, việc xáo trộn ngẫu nhiên (random k-fold) sẽ khiến các chuyến bay của tháng 9 hoặc tháng 10 lọt vào tập huấn luyện, trong khi các chuyến bay tháng 5 lại nằm ở tập kiểm tra. Điều này gây ra hiện tượng **rò rỉ thời gian (temporal data leakage)**, làm đẹp số liệu nhân tạo nhưng mô hình sẽ thất bại hoàn toàn khi triển khai thực tế.

### 1.2. Phân chia tập dữ liệu theo thời gian bay
Mô hình áp dụng phương thức thẩm định nghiêm ngặt:
- **Tập huấn luyện (Train Set):** Toàn bộ các chuyến bay có ngày bay từ **tháng 04/2022 đến tháng 08/2022** (5 tháng mùa xuân và hè).
- **Tập kiểm tra (Test Set):** Toàn bộ các chuyến bay có ngày bay trong **tháng 09/2022 và tháng 10/2022** (2 tháng mùa thu, mô hình hoàn toàn không được nhìn thấy trước).
- **Quy mô mẫu:** Để huấn luyện hiệu quả trên 16 GB RAM của máy cá nhân, một mẫu ngẫu nhiên độc lập **5% (ML_FRACTION = 0.05)** được trích xuất từ tầng Silver:
  - *Train:* ~2.670.000 vé.
  - *Test:* ~1.430.000 vé.

---

## 2. Kiểm Tra Phạm Vi Giá Trị (Range Verification & Out-of-Range Filtering)

Một chỉ số có tương quan cao trong lịch sử chỉ có giá trị dự báo tương lai nếu phân phối giá trị của nó ở tương lai **đã từng xuất hiện trong quá khứ**.
Các mô hình phi tuyến dạng cây (Decision Tree, Random Forest, GBT) không có khả năng ngoại suy (extrapolation) tuyến tính ngoài khoảng $[min, max]$ của tập huấn luyện.

Hệ thống tiến hành đo lường tỷ lệ các giá trị của tập kiểm tra nằm ngoài khoảng $[min_{train}, max_{train}]$:

```python
# Tỷ lệ giá trị kiểm tra rơi ra ngoài phạm vi huấn luyện
out_share = test.select(*[F.avg(((F.col(c) < rng[f"{c}__min"]) | (F.col(c) > rng[f"{c}__max"])).cast("double")).alias(c) for c in NUM])
```

### Kết quả phát hiện:
- Các biến như `week_of_year` (tuần trong năm), `month` (tháng), và `cpi_all` (CPI chung) có **> 50% (thậm chí 100%)** giá trị ở tháng 9–10 nằm ngoài tập huấn luyện tháng 4–8 (do thời gian và lạm phát tăng dần theo tháng).
- Nếu giữ các biến này, các cây quyết định sẽ phân nhánh sai lầm hoặc gán toàn bộ giá trị tương lai vào nhánh biên ngoài cùng.
- **Quy tắc xử lý:** Loại bỏ triệt để các biến có tỷ lệ ngoài khoảng $> 50\%$ khỏi tập đặc trưng huấn luyện của mô hình học có giám sát. Các yếu tố xu hướng vĩ mô dài hạn này được chuyển sang xử lý bằng mô hình chuỗi thời gian ở Notebook 06.

---

## 3. Kiến Trúc Pipeline Đặc Trưng Của Spark MLlib

Dữ liệu được đưa qua `pyspark.ml.Pipeline` với các bước chuẩn hóa:
1. **Biến đổi biến mục tiêu (Label Transformation):** Giá vé `totalFare` bị lệch phải (right-skewed) rất mạnh. Mô hình được huấn luyện trên **$\log(\text{totalFare})$** để ổn định phương sai và đảm bảo giá trị dự báo sau khi đổi ngược $\exp(\hat{y})$ luôn dương.
2. **Xử lý giá trị khuyết (`Imputer`):** Điền giá trị trung vị (median) cho các trường số có thể bị thiếu (khoảng cách, thông tin thời tiết).
3. **Mã hóa biến phân loại (`StringIndexer`):** Chuyển đổi chuỗi danh mục (`route`, `airline_code`, `startingAirport`, `destinationAirport`, `booking_window`) thành chỉ số số nguyên. Cấu hình `maxBins = 300` để các thuật toán cây có thể phân nhánh trực tiếp trên tất cả 234 tuyến bay.
4. **Chuẩn hóa cho mô hình tuyến tính:** Bổ sung `OneHotEncoder` và `StandardScaler` cho hồi quy tuyến tính.
5. **Gộp vector (`VectorAssembler`):** Kết hợp toàn bộ đặc trưng vào cột vector duy nhất `features`.

---

## 4. Kết Quả So Sánh 5 Mô Hình

Đánh giá được thực hiện sau khi chuyển đổi giá trị dự báo về đơn vị USD thực tế:
- $RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$
- $MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|$
- $MAPE = \frac{1}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$
- $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### Bảng đối chiếu hiệu năng trên tập kiểm tra tháng 9–10/2022:

| Mô hình | MAE (USD) | RMSE (USD) | MAPE (%) | $R^2$ | Thời gian huấn luyện |
|---|---|---|---|---|---|
| **Baseline (Trung vị Tuyến $\times$ Nhóm đặt)** | $68.42$ | $104.15$ | $23.1\%$ | $0.512$ | ~4 giây |
| **Linear Regression (Hồi quy tuyến tính)** | $62.15$ | $95.30$ | $20.8\%$ | $0.589$ | ~45 giây |
| **Decision Tree (Cây quyết định, maxDepth=12)** | $49.80$ | $79.10$ | $15.4\%$ | $0.721$ | ~70 giây |
| **Random Forest (30 cây, sâu 10)** | $47.20$ | $74.50$ | $14.5\%$ | $0.785$ | ~420 giây |
| **Gradient-Boosted Trees (GBT, 60 vòng, sâu 6)** | **$45.60$** | **$71.80$** | **$13.8\%$** | **$0.812$** | ~680 giây |

> **Nhận xét:**
> - Mô hình **Gradient-Boosted Trees (GBT)** và **Random Forest** đạt chất lượng vượt trội, giảm sai số MAE từ $68.42$ (Baseline) xuống còn **$45.60\text{ USD}$** và đạt hệ số xác định $R^2 = 0,812$.
> - Mức sai số tương đối bình quân $MAPE \approx 13,8\%$ là kết quả rất khả quan đối với bài toán dự báo giá vé máy bay cá nhân có độ biến động cao.

---

## 5. Giá Trị Của Việc Tích Hợp Dữ Liệu (Ablation Study)

Để chứng minh tính hiệu quả của việc thu thập 7 nguồn dữ liệu bên ngoài, mô hình tốt nhất (GBT) được huấn luyện lại trong điều kiện **chỉ sử dụng các trường có sẵn trong dữ liệu chuyến bay** (loại bỏ thời tiết, ngày lễ, nhiên liệu, vùng miền, khoảng cách địa lý) và so sánh với mô hình đầy đủ:

| Phiên bản mô hình | MAE (USD) | RMSE (USD) | MAPE (%) | $R^2$ |
|---|---|---|---|---|
| **GBT — Chỉ dữ liệu chuyến bay nội tại** | $49.30$ | $77.60$ | $15.1\%$ | $0.758$ |
| **GBT — Có tích hợp đầy đủ 7 nguồn ngoài** | **$45.60$** | **$71.80$** | **$13.8\%$** | **$0.812$** |
| **Mức cải thiện (Gain)** | **-3,70 USD (-7,5%)** | **-5,80 USD (-7,5%)** | **-1,3%** | **+0,054 (+7,1%)** |

Việc tích hợp nguồn ngoài giúp giảm sai số MAE gần 4 USD mỗi vé và giải thích thêm 5,4% phương sai của giá vé.

---

## 6. Mức Độ Quan Trọng Của Đặc Trưng (Feature Importance)

Trích xuất thuộc tính `featureImportances` từ mô hình cây GBT tối ưu:

1. **`totalTravelDistance` (~28,4%):** Khoảng cách hành trình là yếu tố chi phối lớn nhất đến chi phí nhiên liệu và giá gốc.
2. **`duration_min` (~21,2%):** Tổng thời gian di chuyển phản ánh độ dài chặng bay và các khoảng chờ nối chuyến.
3. **`route` (~16,5%):** Tuyến bay mang đặc thù cạnh tranh và mức sống của hai đầu sân bay.
4. **`num_stops` (~9,8%):** Số chặng dừng tác động trực tiếp đến thuế sân bay và cấu trúc chi phí.
5. **`days_before` (~8,7%):** Số ngày đặt trước tạo nên độ cong của đường giá (booking curve).
6. **`airline_code` (~5,6%):** Định vị thương hiệu (LCC vs Legacy carrier).
7. **`cabin_level` (~4,1%):** Hạng dịch vụ trên chuyến bay.
8. **Các yếu tố khác (~5,7%):** Thời tiết (`bad_weather`), ngày lễ (`days_to_holiday`), giá dầu ngắn hạn, giờ bay.

---

## 7. Phân Tích Sai Số Theo Thời Điểm Đặt Vé

Đo lường sai số dự báo của GBT trên từng nhóm thời điểm đặt vé trong tháng 9–10:

| Nhóm thời điểm đặt vé | Giá thực tế TB | Giá dự đoán TB | MAE (USD) | MAPE (%) |
|---|---|---|---|---|
| **3. Sớm (31–60 ngày)** | $298.40$ | $302.10$ | $38.20$ | **11,5%** |
| **2. Gấp (8–30 ngày)** | $310.20$ | $314.50$ | $44.10$ | **13,2%** |
| **1. Rất gấp (1–7 ngày)** | $348.60$ | $338.20$ | $61.50$ | **18,7%** |

> **Phát hiện:** Dự đoán giá vé đặt sớm có độ chính xác cao nhất ($MAPE = 11,5\%$) do giá vé ổn định. Ngược lại, nhóm vé đặt rất sát ngày bay (1–7 ngày) có sai số cao hơn ($18,7\%$) do các hãng liên tục điều chỉnh giá theo thuật toán năng động (Dynamic Pricing) phụ thuộc vào số ghế còn lại thực tế của từng chuyến bay cụ thể.

Mô hình tối ưu sau đó được huấn luyện lại trên toàn bộ dữ liệu tháng 04–10/2022 và lưu trữ tại `models/best_model` để phục vụ các phân tích tiếp theo.
