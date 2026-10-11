# Phân Tích Khám Phá & Ma Trận Tương Quan (Tầng Gold)

Tài liệu này trình bày các phát hiện quy luật giá vé trên toàn bộ 82,1 triệu dòng dữ liệu (tầng Gold tại [04_gold_analysis_correlation.py](file:///f:/PKA/Code/BD/flight-price-analysis/notebooks/04_gold_analysis_correlation.py)), hành vi định giá theo thời điểm đặt vé (Booking Curve & Urgency Premium), và cơ chế lựa chọn đặc trưng toán học cho mô hình Machine Learning.

---

## 1. Các Chỉ Số Tổng Quan (KPIs Trên Toàn Bộ Dữ Liệu)

Các phép tính tổng hợp được Spark thực thi trên toàn bộ **82.137.463 bản ghi Silver**:

| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| **Tổng số vé phân tích** | `82.137.463` | Số lượt báo giá vé máy bay hợp lệ |
| **Số hành trình duy nhất (`legId`)** | `5.999.739` | Gần 6 triệu chuyến bay vật lý khác nhau |
| **Số tuyến bay (`route`)** | `234` tuyến | Kết nối có hướng giữa 16 sân bay trung tâm lớn |
| **Số hãng hàng không** | `14` hãng | Bao gồm cả hãng truyền thống (FSC), giá rẻ (LCC/ULCC), và khu vực |
| **Giá vé trung bình (`avg_fare`)** | **$340.39** | Bị kéo lên bởi nhóm vé sát giờ bay và hạng vé thương gia |
| **Giá vé trung vị (`median_fare`)** | **$305.20** | Phản ánh mức giá phổ biến nhất của thị trường |
| **Thời gian tìm kiếm** | `16/04/2022` $\to$ `05/10/2022` | Khoảng thời gian người dùng tra cứu |
| **Thời gian bay** | `17/04/2022` $\to$ `19/11/2022` | Giai đoạn cất cánh |

---

## 2. Quy Luật Đặt Chỗ: Đặt Sớm Hay Đi Gấp?

### 2.1. Phân bố theo 3 nhóm thời điểm đặt vé (`booking_window`)
So sánh với nhóm khách hàng chủ động đặt sớm (31–60 ngày trước ngày bay):

| Nhóm thời điểm | Số vé | Tỷ lệ (%) | Giá TB | Giá Trung vị | Chênh lệch giá trung vị |
|---|---|---|---|---|---|
| **3. Sớm (31–60 ngày)** | 34.464.006 | 42,0% | $328.97 | **$288.10** | *Mốc tham chiếu (0%)* |
| **2. Gấp (8–30 ngày)** | 35.732.869 | 43,5% | $341.74 | **$305.61** | **+6,1%** (+$17.51) |
| **1. Rất gấp (1–7 ngày)** | 11.940.588 | 14,5% | $369.30 | **$342.60** | **+18,9%** (+$54.50) |

### 2.2. Điểm gãy đường cong giá vé (Booking Curve Inflection Point)
Khảo sát đường giá vé chi tiết từ ngày 60 lùi về ngày 1 trước khi bay cho thấy:
- **Từ ngày 60 đến ngày 21:** Giá vé tương đối ổn định, dao động trong dải trung vị $280 – $300.
- **Từ ngày 21 (3 tuần trước bay):** Bắt đầu xuất hiện xu hướng tăng giá khi các hạng đặt chỗ giá rẻ (fare classes) được các hãng hàng không đóng dần theo thuật toán Revenue Management.
- **Từ ngày 7 đến ngày 1:** Giá vé tăng vọt theo phương thẳng đứng. Giá trung vị chạm đỉnh $361.60 ở ngày 1 (tăng hơn **25,5%** so với mức đáy ở ngày 45 là $274.73).

---

## 3. Mức Độ Đội Giá Khi Đi Gấp (Urgency Premium Analysis)

Phép đo Urgency Premium được tính bằng tỷ lệ phần trăm chênh lệch giá trung vị giữa nhóm **Rất gấp (1–7 ngày)** và nhóm **Đặt sớm (31–60 ngày)**.

### 3.1. Phân hóa theo mô hình hãng hàng không
Thuật toán định giá phân hóa cực kỳ rõ rệt giữa hai nhóm: Hãng truyền thống (Legacy) và Hãng siêu tiết kiệm (ULCC):

| Hãng hàng không | Phân loại | Giá sớm (31–60d) | Giá gấp (1–7d) | Mức đội giá (%) | Đặc điểm kinh doanh |
|---|---|---|---|---|---|
| **Southern Airways** | Commuter | $257.00 | $468.50 | **+82,3%** | Chặng bay ngắn kết nối vùng |
| **Spirit Airlines** | ULCC | $163.58 | $271.58 | **+66,0%** | Giá vé sàn cực rẻ, phạt nặng người mua gấp |
| **Frontier Airlines** | ULCC | $152.99 | $213.99 | **+39,9%** | Tương tự Spirit, bóc tách triệt để dịch vụ |
| **JetBlue Airways** | Hybrid | $215.60 | $270.10 | **+25,3%** | Phục vụ kết hợp khách du lịch và thương mại |
| **American Airlines**| Legacy | $257.61 | $304.11 | **+18,1%** | Mạng lưới bay rộng, giá sàn vừa phải |
| **Delta Air Lines** | Legacy | $357.20 | $394.11 | **+10,3%** | Giá sàn cao, biên độ tăng khi mua gấp thấp |
| **United Airlines** | Legacy | $347.61 | $358.60 | **+3,2%** | Giá sàn cao, giữ giá ổn định cho khách doanh nghiệp |

> **Quy luật rút ra:** Các hãng hàng không giá rẻ có tỷ lệ tăng giá khi mua gấp cao gấp **4 đến 10 lần** so với các hãng truyền thống lớn.

### 3.2. Tuyến bay đội giá mạnh nhất và tuyến giữ giá
- **Top tuyến đội giá cao nhất (> 90% khi mua gấp):**
  - `CLT-DTW` / `DTW-CLT` (+113,8% và +112,2%): Tuyến nối 2 trung tâm công nghiệp lớn, lưu lượng khách công tác đột xuất cao.
  - `DTW-ATL` / `ATL-DTW` (+104,3% và +98,6%): Trục bay chính kết nối trung tâm Delta.
  - `PHL-JFK` / `PHL-DTW` (+99,0% và +85,7%): Chặng ngắn bờ Đông.
- **Top tuyến giữ giá hoặc giảm nhẹ:**
  - `BOS-IAD` (-28,6%), `JFK-DFW` (-28,1%), `BOS-ORD` (-21,7%): Các tuyến đường dài cạnh tranh khốc liệt giữa nhiều hãng bay, hoặc có tần suất bay dày đặc khiến các chuyến bay giờ chót còn dư ghế phải xả vé rẻ để lấp đầy chỗ.

---

## 4. Các Yếu Tố Khác Tác Động Tới Giá Vé

1. **Thứ trong tuần (`day_of_week`):**
   - Đắt nhất: **Chủ nhật** (trung vị $367.60) và **Thứ 6** ($327.11) do nhu cầu đi lại cuối tuần.
   - Rẻ nhất: **Thứ 3** (trung vị $252.60) và **Thứ 4** ($258.60) — rẻ hơn Chủ nhật tới **31%**.
2. **Khung giờ khởi hành (`dep_hour`):**
   - Đắt nhất: Khung sáng sớm 06:00–08:00 ($311 – $321) và đêm muộn 22:00–23:00 ($351 – $378).
   - Rẻ nhất: Khung chiều tối 19:00–20:00 (trung vị $248 – $252).
3. **Số điểm dừng (`num_stops`):**
   - Bay thẳng (`0 stops`, 22,1M vé): Giá trung vị **$215.60**.
   - 1 điểm dừng (`1 stop`, 52,3M vé): Giá trung vị **$318.69** (+47,8%).
   - 2 điểm dừng (`2 stops`, 7,6M vé): Giá trung vị **$540.70** (+150,8%).
   *(Nối chuyến đắt hơn bay thẳng vì đa số là các chặng xuyên bang đường dài hoặc bay vào các sân bay phụ).*
4. **Hiệu ứng ngày lễ (`days_to_holiday`):**
   - Ngày đúng dịp lễ (`0`): Giá trung vị vọt lên **$358.60**.
   - Ngày trước lễ 6 ngày (`-6`): Đạt đỉnh **$385.20** do hành khách bắt đầu di chuyển sớm.

---

## 5. Ma Trận Tương Quan & Tiêu Chí Lựa Chọn Đặc Trưng

Hệ thống tính toán ma trận tương quan giữa giá vé (`totalFare`), log giá vé (`log_fare`) và 32 đặc trưng số trên toàn bộ 82M dòng bằng `pyspark.ml.stat.Correlation`:

### 5.1. Hệ số tương quan Pearson & Spearman (Top biến số)

| Đặc trưng | Pearson với `totalFare` | Pearson với `log(fare)` | Spearman với `totalFare` | Bản chất tương quan |
|---|---|---|---|---|
| `totalTravelDistance` | **+0,4926** | **+0,5231** | **+0,5232** | Tuyến bay càng dài tiêu tốn càng nhiều nhiên liệu & chi phí |
| `duration_min` | **+0,4791** | **+0,5103** | **+0,5166** | Thời gian hành trình bay |
| `num_stops` | **+0,4138** | **+0,4428** | **+0,4570** | Chặng bay nối chuyến |
| `non_stop` | **-0,2981** | **-0,3145** | **-0,3230** | Bay thẳng thường là chặng ngắn hơn nối chuyến |
| `cabin_level` | **+0,1984** | **+0,2140** | **+0,2210** | Hạng thương gia/nhất đắt hơn phổ thông |
| `basic_economy` | **-0,1942** | **-0,2180** | **-0,2240** | Vé tiết kiệm cơ bản rẻ hơn vé thường |
| `lat_diff` | **+0,1652** | **+0,1720** | **+0,1780** | Chênh lệch vĩ độ bắc-nam |
| `cross_region` | **+0,1432** | **+0,1510** | **+0,1540** | Bay liên vùng |
| `days_before` | **-0,1187** | **-0,1320** | **-0,1380** | Đặt càng xa ngày bay thì giá càng thấp |
| `seatsRemaining` | **+0,0981** | **+0,1050** | **+0,1090** | Số ghế còn lại hiển thị |
| `is_weekend` | **+0,0720** | **+0,0750** | **+0,0760** | Ngày bay cuối tuần |

### 5.2. Tỷ số tương quan phương sai $\eta$ (Eta) cho biến phân loại
$\eta$ đo lường mức độ biến thiên của giá vé được giải thích bởi nhóm:
- `route` (Tuyến bay): **$\eta = 0,3341$** $\to$ Yếu tố phân loại quan trọng nhất.
- `airline_code` (Hãng bay): **$\eta = 0,2452$** $\to$ Mỗi hãng có chính sách giá riêng.
- `startingAirport` (Sân bay đi): **$\eta = 0,1460$**.
- `destinationAirport` (Sân bay đến): **$\eta = 0,1420$**.
- `origin_region` / `dest_region`: **$\eta = 0,1120$**.

### 5.3. Tiêu chí chọn đặc trưng cho mô hình ML (`feature_selection.json`)
Thuật toán chọn biến áp dụng bộ lọc 4 bước:
1. **Ngưỡng tương quan tối thiểu:** Giữ biến số có $\max(|r_{\text{Pearson}}|, |r_{\text{Spearman}}|) \ge 0,05$. Giữ biến phân loại có $\eta \ge 0,10$.
2. **Luôn giữ trục bài toán:** `days_before` luôn được giữ bất kể tương quan vì đây là biến cốt lõi cho phân tích giá theo ngày đặt.
3. **Loại bỏ đa cộng tuyến ($|r| > 0,9$):** Ví dụ: `duration_min` và `totalTravelDistance` có tương quan với nhau $r = 0,88$ (được giữ cả hai), nhưng nếu hai biến đo lường trùng lặp $> 0,9$, biến có tương quan với giá vé yếu hơn sẽ bị loại bỏ.
4. **Loại bỏ rò rỉ đáp án (Data Leakage):** Loại bỏ `fareBasisCode` mặc dù $\eta = 0,52$ rất cao, vì mã giá vé chứa sẵn ký tự định giá của hãng hàng không, tương đương với việc biết trước đáp án.

Kết quả lưu tại `gold/feature_selection.json` cung cấp danh sách đặc trưng chính thức cho Notebook 05.
