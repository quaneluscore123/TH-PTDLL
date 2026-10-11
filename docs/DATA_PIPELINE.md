# Pipeline Dữ Liệu: Ingestion, 7 Nguồn Ngoài & Tầng Silver

Tài liệu này chi tiết hóa quy trình thu thập dữ liệu thô, tích hợp 7 nguồn thông tin bên ngoài và quy trình chuyển hóa dữ liệu tại tầng Silver (57 cột đặc trưng).

---

## 1. Nguồn Dữ Liệu Gốc (Raw Flight Ingestion)

Tệp dữ liệu gốc `itineraries.csv` được thu thập từ nền tảng Expedia thông qua bộ dữ liệu công khai Kaggle (*dilwong/flightprices*):
- **Quy mô:** 31,0 GB (33.310.142.062 bytes), 82.138.753 dòng.
- **Thời gian tra cứu:** 16/04/2022 đến 05/10/2022.
- **Thời gian bay:** 17/04/2022 đến 19/11/2022 (tập trung khoảng 7 tháng).
- **Phạm vi địa lý:** 16 sân bay trung tâm lớn nhất nước Mỹ:
  `ATL` (Atlanta), `BOS` (Boston), `CLT` (Charlotte), `DEN` (Denver), `DFW` (Dallas/Fort Worth), `DTW` (Detroit), `EWR` (Newark), `IAD` (Washington Dulles), `JFK` (New York JFK), `LAX` (Los Angeles), `LGA` (New York LaGuardia), `MIA` (Miami), `OAK` (Oakland), `ORD` (Chicago O'Hare), `PHL` (Philadelphia), `SFO` (San Francisco).
- **Cấu trúc bản ghi thô (27 cột):** Bao gồm mã hành trình (`legId`), ngày tìm, ngày bay, sân bay đi/đến, giá vé cơ sở (`baseFare`), tổng giá vé (`totalFare`), số ghế còn lại (`seatsRemaining`), khoảng cách bay (`totalTravelDistance`), và các chuỗi gộp ngăn cách bởi ký tự `||` ghi nhận thông tin từng chặng bay (`segments...`).

Tại [01_bronze_ingest.py](file:///f:/PKA/Code/BD/flight-price-analysis/notebooks/01_bronze_ingest.py), toàn bộ 82,1 triệu dòng được nạp thẳng vào Apache Spark với schema cố định và lưu trữ dưới dạng Parquet nén **zstd** (giảm xuống 4,5 GB).

---

## 2. 7 Nguồn Dữ Liệu Bổ Trợ (External Enrichment Sources)

Do bộ dữ liệu Expedia chỉ phản ánh 7 tháng trong năm 2022 và thiếu các yếu tố ngoại vi tác động tới giá vé, dự án thu thập và chuẩn hóa thêm 7 nguồn dữ liệu bên ngoài tại [02_external_sources.py](file:///f:/PKA/Code/BD/flight-price-analysis/notebooks/02_external_sources.py):

| # | Nguồn | Cấp độ chi tiết | Mục đích sử dụng |
|---|---|---|---|
| **1** | **OpenFlights** | Sân bay (16 IATA) | Cung cấp tên thành phố, bang, tọa độ địa lý (`lat`, `lon`), độ cao và phân vùng địa lý (East, Central, Mountain, West). |
| **2** | **Thư viện `holidays`** | Ngày (2022–2027) | Danh mục ngày lễ liên bang Mỹ. Dùng để tính biến cờ `is_holiday` và khoảng cách ngày tới ngày lễ gần nhất (`days_to_holiday`). |
| **3** | **Open-Meteo Historical API** | Sân bay $\times$ Ngày bay | Dữ liệu thời tiết lịch sử theo ngày cho 16 sân bay từ 01/04 đến 30/11/2022: nhiệt độ trung bình, nhiệt độ cao nhất, lượng mưa (mm), tuyết rơi (cm), tốc độ gió tối đa, biến cờ thời tiết xấu `bad_weather`. |
| **4** | **EIA (U.S. Energy Information Admin)** | Ngày & Tháng (1990–2026) | Giá nhiên liệu phản lực (Kerosene-Type Jet Fuel, U.S. Gulf Coast, USD/gallon). Nhiên liệu chiếm 25–35% tổng chi phí vận hành hàng không. |
| **5** | **BLS (Bureau of Labor Statistics)** | Tháng (2005–2026) | Chỉ số giá tiêu dùng: `CUUR0000SETG01` (CPI vé máy bay) và `CUUR0000SA0` (CPI chung/lạm phát). Dùng làm chuỗi thời gian để đo lường xu hướng vĩ mô dài hạn. |
| **6** | **DOT Consumer Airfare Report (Table 1a)** | Tuyến bay $\times$ Quý (1993–Q1/2024) | Giá vé thực tế bình quân một chiều theo cặp sân bay (Bộ Giao thông Vận tải Mỹ). Là **chân lý mặt đất (Ground Truth)** để kiểm chứng mô hình dự báo 2023. |
| **7** | **BTS Average Domestic Itinerary Fare** | Sân bay $\times$ Quý (1993–Q3/2024) | Giá vé bình quân thực tế theo từng sân bay phục vụ phân tích đối chiếu. |

Tất cả 7 bảng sau khi làm sạch được lưu trữ tại `external/clean/*.csv` với dung lượng nhỏ (tổng cộng < 5 MB), sẵn sàng nạp vào Spark memory.

---

## 3. Quy Trình Chuyển Hóa Tầng Silver (03_silver_etl.py)

Tầng Silver tại [03_silver_etl.py](file:///f:/PKA/Code/BD/flight-price-analysis/notebooks/03_silver_etl.py) thực thi các tác vụ:

### 1. Quy tắc làm sạch (Data Cleaning)
- Bỏ các bản ghi có `totalFare <= 0` (giá vé không hợp lệ).
- Bỏ các bản ghi có `searchDate IS NULL` hoặc `flightDate IS NULL`.
- Bỏ các bản ghi có ngày bay trước ngày tìm (`datediff(flightDate, searchDate) < 0`).
- Khử trùng lặp trên bộ khóa `[legId, searchDate, fareBasisCode, totalFare]`.

### 2. Kỹ nghệ đặc trưng chuyến bay (Feature Engineering)
- **Khoảng cách đặt vé:** `days_before = datediff(flightDate, searchDate)`.
- **Nhóm thời điểm đặt vé (`booking_window`):**
  - `1. Rất gấp (1-7 ngày)`: Hành khách đi công tác đột xuất hoặc tình huống khẩn cấp.
  - `2. Gấp (8-30 ngày)`: Kế hoạch ngắn hạn.
  - `3. Sớm (31-60 ngày)`: Kế hoạch du lịch / nghỉ dưỡng đặt trước.
- **Thời gian bay:** Chuyển đổi định dạng ISO 8601 `travelDuration` (vd: `PT2H29M`, `P1DT3H5M`) thành số phút tổng cộng `duration_min`.
- **Thông tin chặng:** Tách phần tử đầu tiên của chuỗi `segments...` để lấy hãng hàng không chính (`airline_code`, `airline_name`), xác định số điểm dừng (`num_stops`), cờ bay nhiều hãng phối hợp (`multi_airline`).
- **Phân cấp hạng ghế (`cabin_level`):** Mã hóa thứ bậc hạng vé: `coach` (1), `premium coach` (2), `business` (3), `first` (4).
- **Thời gian & lịch trình:** Trích xuất giờ khởi hành (`dep_hour`), thứ trong tuần của ngày bay (`day_of_week`), thứ trong tuần khi tìm vé (`search_dow`), cờ cuối tuần (`is_weekend`).
- **Đơn giá dặm bay:** `fare_per_mile = totalFare / totalTravelDistance`.

### 3. Tích hợp nguồn ngoài (Broadcast Joins)
- **Sân bay:** Ghép theo `startingAirport` và `destinationAirport` để lấy tọa độ, thành phố, vùng địa lý (`origin_region`, `dest_region`), tính độ chênh vĩ độ (`lat_diff`) và cờ bay liên vùng (`cross_region`).
- **Ngày lễ:** Ghép theo `flightDate` lấy cờ `is_holiday`, số ngày tới ngày lễ (`days_to_holiday`) và cửa sổ ngày lễ ($\pm 3$ ngày quanh dịp lễ: `holiday_window`).
- **Thời tiết:** Ghép theo `[startingAirport, flightDate]` và `[destinationAirport, flightDate]` lấy nhiệt độ, mưa, cờ thời tiết xấu nơi đi (`origin_bad_weather`) và nơi đến (`dest_bad_weather`).
- **Giá nhiên liệu:** Ghép theo `searchDate` lấy giá dầu phiên gần nhất (`jet_fuel_usd_gal`) và tỷ lệ biến động 7 ngày (`fuel_7d_change`).
- **Lạm phát & CPI:** Ghép theo tháng tìm vé (`search_month`) lấy chỉ số `cpi_all`, `cpi_airfare`, `inflation_yoy`, `airfare_yoy`.

---

## 4. Tỷ Lệ Ghép Nối Nguồn Ngoài (Join Match Rate)

Kết quả đo đạc tỷ lệ dữ liệu ghép nối thành công trên toàn bộ 82,1 triệu dòng Silver:

| Chỉ số tích hợp | Trường dữ liệu | Tỷ lệ ghép thành công (%) | Ghi chú |
|---|---|---|---|
| Thông tin sân bay đi | `origin_city` | **100,00%** | Toàn bộ 16 sân bay có trong OpenFlights |
| Thông tin sân bay đến | `dest_city` | **100,00%** | Toàn bộ 16 sân bay có trong OpenFlights |
| Lịch ngày lễ | `days_to_holiday` | **100,00%** | Bao phủ đầy đủ từ 01/03 đến 31/12/2022 |
| Thời tiết nơi đi | `origin_temp_mean` | **100,00%** | Khớp 100% tọa độ 16 sân bay $\times$ ngày bay |
| Thời tiết nơi đến | `dest_temp_mean` | **100,00%** | Khớp 100% tọa độ 16 sân bay $\times$ ngày bay |
| Giá nhiên liệu bay | `jet_fuel_usd_gal` | **100,00%** | Forward-fill các ngày cuối tuần và nghỉ lễ |
| Lạm phát / CPI | `cpi_all` | **100,00%** | Nội suy và khớp trọn vẹn theo tháng tìm |

---

## 5. Từ Điển Dữ Liệu Tầng Silver (Silver Schema - 57 Cột)

Dữ liệu Silver được lưu trữ tại `silver/itineraries` phân vùng theo `flight_month`:

| Nhóm đặc trưng | Tên cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|---|
| **Định danh & Ngày** | `legId` | String | Mã duy nhất của hành trình |
| | `searchDate` | Date | Ngày tìm kiếm vé trên Expedia |
| | `flightDate` | Date | Ngày máy bay cất cánh |
| | `flight_month` | String | Phân vùng (Partition Key): `YYYY-MM` |
| | `search_month` | String | Tháng thực hiện tìm kiếm: `YYYY-MM` |
| **Lộ trình & Sân bay** | `startingAirport` | String | Mã IATA sân bay xuất phát (3 ký tự) |
| | `destinationAirport`| String | Mã IATA sân bay hạ cánh (3 ký tự) |
| | `route` | String | Tuyến bay có hướng: `ORIGIN-DEST` (vd: `JFK-LAX`) |
| | `origin_city`, `dest_city` | String | Tên thành phố đi và đến |
| | `origin_region`, `dest_region`| String | Vùng địa lý: East, Central, Mountain, West |
| | `origin_lat`, `dest_lat` | Double | Vĩ độ sân bay xuất phát và hạ cánh |
| | `lat_diff` | Double | Khoảng cách vĩ độ tuyệt đối giữa 2 sân bay |
| | `cross_region` | Integer | Cờ bay xuyên vùng (1: khác vùng, 0: cùng vùng) |
| **Giá vé & Kinh tế** | `baseFare` | Double | Giá vé cơ sở chưa thuế phí (USD) |
| | `totalFare` | Double | **Biến mục tiêu (Label):** Tổng giá vé (USD) |
| | `fare_per_mile` | Double | Đơn giá trên mỗi dặm bay (USD/mile) |
| | `fareBasisCode` | String | Mã định giá vé của hãng hàng không |
| **Hành trình chi tiết**| `duration_min` | Integer | Tổng thời gian bay và chuyển tiếp (phút) |
| | `num_stops` | Integer | Số chặng dừng (0: bay thẳng, 1: nối chuyến...) |
| | `non_stop` | Integer | Cờ bay thẳng (1: bay thẳng, 0: có dừng) |
| | `airline_code` | String | Mã IATA hãng hàng không chính (vd: `AA`, `DL`) |
| | `airline_name` | String | Tên đầy đủ của hãng hàng không |
| | `multi_airline` | Integer | Cờ chuyến bay kết hợp nhiều hãng khác nhau |
| | `cabin_level` | Integer | Hạng ghế (1: coach, 2: premium, 3: business, 4: first) |
| | `basic_economy` | Integer | Cờ vé hạng tiết kiệm cơ bản (Basic Economy) |
| | `refundable` | Integer | Cờ vé cho phép hoàn hủy |
| | `seatsRemaining` | Integer | Số ghế còn trống hiển thị trên hệ thống |
| | `totalTravelDistance`| Double | Tổng quãng đường hành trình bay (dặm) |
| | `elapsedDays` | Integer | Số ngày chuyển tiếp qua múi giờ / qua đêm |
| **Thời điểm & Thói quen**| `days_before` | Integer | **Trục chính:** Số ngày đặt vé trước ngày bay |
| | `booking_window` | String | Nhóm đặt vé: 1-7 ngày, 8-30 ngày, 31-60 ngày |
| | `dep_hour` | Integer | Giờ cất cánh theo giờ địa phương (0–23) |
| | `day_of_week` | Integer | Thứ bay trong tuần (1 = CN, 2 = T2 ... 7 = T7) |
| | `search_dow` | Integer | Thứ trong tuần khi thực hiện tìm kiếm |
| | `is_weekend` | Integer | Cờ ngày bay rơi vào cuối tuần (Thứ 7 / Chủ nhật) |
| | `month` | Integer | Tháng bay (4 đến 11) |
| | `week_of_year` | Integer | Số thứ tự tuần trong năm (1 đến 52) |
| **Ngày lễ Mỹ** | `is_holiday` | Integer | Cờ ngày bay trùng ngày lễ liên bang |
| | `days_to_holiday` | Integer | Số ngày tới ngày lễ gần nhất (âm: sau lễ, dương: trước lễ) |
| | `holiday_window` | Integer | Cờ bay trong vòng $\pm 3$ ngày quanh ngày lễ |
| **Thời tiết lịch sử** | `origin_temp_mean` | Double | Nhiệt độ bình quân ngày bay tại nơi đi (°C) |
| | `origin_precip_mm` | Double | Lượng mưa tích lũy trong ngày tại nơi đi (mm) |
| | `origin_snow_cm` | Double | Tuyết rơi tại nơi đi (cm) |
| | `origin_wind_max` | Double | Tốc độ gió giật tối đa tại nơi đi (km/h) |
| | `origin_bad_weather`| Integer| Cờ thời tiết xấu nơi đi (mưa $\ge 10\text{mm}$, tuyết $\ge 1\text{cm}$, gió $\ge 40$) |
| | `dest_temp_mean` | Double | Nhiệt độ bình quân tại sân bay đến (°C) |
| | `dest_precip_mm` | Double | Lượng mưa tích lũy tại sân bay đến (mm) |
| | `dest_bad_weather` | Integer| Cờ thời tiết xấu tại sân bay đến |
| **Kinh tế vĩ mô** | `jet_fuel_usd_gal` | Double | Giá dầu Jet Fuel ngày tìm vé (USD/gallon) |
| | `fuel_7d_change` | Double | Tỷ lệ thay đổi giá dầu trong 7 ngày (%) |
| | `cpi_all` | Double | Chỉ số CPI chung tháng tìm vé |
| | `cpi_airfare` | Double | Chỉ số CPI giá vé máy bay tháng tìm vé |
| | `inflation_yoy` | Double | Tỷ lệ lạm phát so với cùng kỳ năm trước (%) |
| | `airfare_yoy` | Double | Tỷ lệ biến động CPI vé máy bay so với cùng kỳ (%) |
