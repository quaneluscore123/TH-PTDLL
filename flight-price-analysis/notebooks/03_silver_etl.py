# %% [markdown]
# # 03 — ETL: làm sạch, tạo đặc trưng và tích hợp dữ liệu (tầng Silver)
#
# **Mốc pipeline:** Bronze → **Silver** — chạy trên **toàn bộ 82 triệu dòng** bằng Spark.
#
# 1. Làm sạch: bỏ giá ≤ 0, ngày trống, ngày bay trước ngày tìm; bỏ dòng trùng lặp.
# 2. Tạo đặc trưng từ chuyến bay: số ngày đặt trước, nhóm đặt vé (sớm / gấp / rất gấp), thời lượng bay,
#    số điểm dừng, hãng, hạng ghế, giờ khởi hành, thứ, tháng...
# 3. Tích hợp nguồn ngoài (broadcast join vì các bảng đều nhỏ):
#    sân bay · ngày lễ · thời tiết nơi đi và nơi đến · giá nhiên liệu theo ngày · CPI và lạm phát theo tháng.
# 4. Ghi Parquet, **phân vùng theo tháng bay** (`flight_month`) để các bước sau lọc nhanh theo thời gian.

# %%
import sys, os, time
sys.path.insert(0, os.path.abspath(".."))
from common import *
import pandas as pd
from pyspark.sql import functions as F

spark = get_spark("03-silver")
CLEAN = f"{EXTERNAL}/clean"
bronze = spark.read.parquet(BRONZE)
n_bronze = bronze.count()
print(f"Bronze: {n_bronze:,} dòng")

# %% [markdown]
# ## 1–2. Làm sạch và tạo đặc trưng từ dữ liệu chuyến bay

# %%
def seg(c, i=0):
    """Lấy phần tử thứ i của cột nhiều chặng, ví dụ 'AA||DL' → 'AA'."""
    return F.split(F.col(c), r"\|\|").getItem(i)

def iso_part(pattern):
    # travelDuration dạng ISO 8601: 'PT2H29M', 'P1DT3H5M'
    return F.coalesce(F.regexp_extract("travelDuration", pattern, 1).try_cast("int"), F.lit(0))

cabin_rank = F.create_map(*[F.lit(x) for x in ["coach", 1, "premium coach", 2, "business", 3, "first", 4]])

flights = (
    bronze
    .filter((F.col("totalFare") > 0) & F.col("searchDate").isNotNull() & F.col("flightDate").isNotNull())
    .withColumn("days_before", F.datediff("flightDate", "searchDate"))
    .filter(F.col("days_before") >= 0)
    # --- thời điểm đặt vé ---
    .withColumn("booking_window", F.when(F.col("days_before") <= 7, "1. Rất gấp (1-7 ngày)")
                                   .when(F.col("days_before") <= 30, "2. Gấp (8-30 ngày)")
                                   .otherwise("3. Sớm (31-60 ngày)"))
    .withColumn("search_dow", F.dayofweek("searchDate"))              # 1 = Chủ nhật … 7 = Thứ bảy
    # --- thời điểm bay ---
    .withColumn("flight_month", F.date_format("flightDate", "yyyy-MM"))
    .withColumn("month", F.month("flightDate"))
    .withColumn("week_of_year", F.weekofyear("flightDate"))
    .withColumn("day_of_week", F.dayofweek("flightDate"))
    .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
    .withColumn("dep_hour", F.substring(seg("segmentsDepartureTimeRaw"), 12, 2).try_cast("int"))
    # --- hành trình ---
    .withColumn("route", F.concat_ws("-", "startingAirport", "destinationAirport"))
    .withColumn("duration_min", iso_part(r"P(\d+)D") * 1440 + iso_part(r"(\d+)H") * 60 + iso_part(r"(\d+)M"))
    .withColumn("num_stops", F.size(F.split("segmentsAirlineCode", r"\|\|")) - 1)
    .withColumn("airline_code", seg("segmentsAirlineCode"))
    .withColumn("airline_name", seg("segmentsAirlineName"))
    .withColumn("multi_airline", (F.size(F.array_distinct(F.split("segmentsAirlineCode", r"\|\|"))) > 1).cast("int"))
    .withColumn("cabin_level", F.array_max(F.transform(F.split("segmentsCabinCode", r"\|\|"), lambda x: cabin_rank[x])))
    .withColumn("basic_economy", F.col("isBasicEconomy").cast("int"))
    .withColumn("refundable", F.col("isRefundable").cast("int"))
    .withColumn("non_stop", F.col("isNonStop").cast("int"))
    .withColumn("fare_per_mile", F.when(F.col("totalTravelDistance") > 0, F.col("totalFare") / F.col("totalTravelDistance")))
    .select("legId", "searchDate", "flightDate", "flight_month", "startingAirport", "destinationAirport", "route",
            "airline_code", "airline_name", "fareBasisCode", "baseFare", "totalFare", "fare_per_mile",
            "days_before", "booking_window", "search_dow", "month", "week_of_year", "day_of_week", "is_weekend",
            "dep_hour", "duration_min", "num_stops", "multi_airline", "cabin_level", "basic_economy",
            "refundable", "non_stop", "seatsRemaining", "totalTravelDistance", "elapsedDays")
    # Bỏ trùng SAU khi đã chọn cột: Spark chỉ phải xáo trộn (shuffle) các cột gọn,
    # không kéo theo các chuỗi segments... dài → ít tốn đĩa tạm hơn nhiều.
    .dropDuplicates(["legId", "searchDate", "fareBasisCode", "totalFare"])
)

# %% [markdown]
# ## 3. Tích hợp các nguồn ngoài

# %%
# --- Sân bay: thành phố, vùng, tọa độ ---
ap = pd.read_csv(f"{CLEAN}/airports.csv")
ap_s = spark.createDataFrame(ap[["iata", "city", "region", "lat", "lon"]])
origin = ap_s.select(F.col("iata").alias("startingAirport"), F.col("city").alias("origin_city"),
                     F.col("region").alias("origin_region"), F.col("lat").alias("origin_lat"))
dest = ap_s.select(F.col("iata").alias("destinationAirport"), F.col("city").alias("dest_city"),
                   F.col("region").alias("dest_region"), F.col("lat").alias("dest_lat"))

# --- Ngày lễ: is_holiday + số ngày tới ngày lễ gần nhất (âm = sau lễ, dương = trước lễ) ---
hol = pd.read_csv(f"{CLEAN}/holidays.csv", parse_dates=["date"])
cal = pd.DataFrame({"flightDate": pd.date_range("2022-03-01", "2022-12-31")})
diff = hol["date"].values[None, :] - cal["flightDate"].values[:, None]
nearest = abs(diff).argmin(axis=1)
cal["days_to_holiday"] = diff[range(len(cal)), nearest].astype("timedelta64[D]").astype(int)
cal["is_holiday"] = cal["flightDate"].isin(hol["date"]).astype(int)
cal["holiday_window"] = (cal["days_to_holiday"].abs() <= 3).astype(int)     # ±3 ngày quanh ngày lễ
cal["flightDate"] = cal["flightDate"].dt.date
hol_s = spark.createDataFrame(cal)

# --- Thời tiết ngày bay tại sân bay đi và sân bay đến ---
w = pd.read_csv(f"{CLEAN}/weather_daily.csv", parse_dates=["date"])
w["date"] = w["date"].dt.date
w_s = spark.createDataFrame(w)
w_origin = w_s.select(F.col("airport").alias("startingAirport"), F.col("date").alias("flightDate"),
                      *[F.col(c).alias(f"origin_{c}") for c in ["temp_mean", "precip_mm", "snow_cm", "wind_max", "bad_weather"]])
w_dest = w_s.select(F.col("airport").alias("destinationAirport"), F.col("date").alias("flightDate"),
                    *[F.col(c).alias(f"dest_{c}") for c in ["temp_mean", "precip_mm", "bad_weather"]])

# --- Giá nhiên liệu bay theo NGÀY TÌM VÉ (cuối tuần/ngày nghỉ lấy giá phiên gần nhất trước đó) ---
fuel = pd.read_csv(f"{CLEAN}/jet_fuel_daily.csv", parse_dates=["date"]).set_index("date")
fuel = fuel.reindex(pd.date_range("2022-01-01", "2022-12-31")).ffill().loc["2022-03-01":]
fuel["fuel_7d_change"] = fuel["jet_fuel_usd_gal"].pct_change(7) * 100
fuel = fuel.rename_axis("searchDate").reset_index()
fuel["searchDate"] = fuel["searchDate"].dt.date
fuel_s = spark.createDataFrame(fuel)

# --- CPI và lạm phát theo THÁNG TÌM VÉ ---
macro = pd.read_csv(f"{CLEAN}/macro_monthly.csv", parse_dates=["date"])
macro = macro[(macro.date >= "2022-01-01") & (macro.date <= "2022-12-01")]
macro["search_month"] = macro["date"].dt.strftime("%Y-%m")
macro_s = spark.createDataFrame(macro[["search_month", "cpi_all", "cpi_airfare", "inflation_yoy", "airfare_yoy"]])

# %%
silver = (
    flights
    .join(F.broadcast(origin), "startingAirport", "left")
    .join(F.broadcast(dest), "destinationAirport", "left")
    .join(F.broadcast(hol_s), "flightDate", "left")
    .join(F.broadcast(w_origin), ["startingAirport", "flightDate"], "left")
    .join(F.broadcast(w_dest), ["destinationAirport", "flightDate"], "left")
    .join(F.broadcast(fuel_s), "searchDate", "left")
    .withColumn("search_month", F.date_format("searchDate", "yyyy-MM"))
    .join(F.broadcast(macro_s), "search_month", "left")
    .withColumn("cross_region", (F.col("origin_region") != F.col("dest_region")).cast("int"))
    .withColumn("lat_diff", F.abs(F.col("origin_lat") - F.col("dest_lat")))
)

t0 = time.time()
silver.write.mode("overwrite").option("compression", "zstd").partitionBy("flight_month").parquet(SILVER)
print(f"Ghi Silver xong sau {(time.time() - t0) / 60:.1f} phút")

# %% [markdown]
# ## Kiểm tra tầng Silver

# %%
silver = spark.read.parquet(SILVER)
n_silver = silver.count()
size_gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(SILVER) for f in fs) / 1e9
print(f"Silver: {n_silver:,} dòng ({n_silver / n_bronze:.2%} của Bronze), {len(silver.columns)} cột, {size_gb:.2f} GB")
print(f"Đã loại {n_bronze - n_silver:,} dòng (giá ≤ 0, thiếu ngày, trùng lặp)")

# %%
by_month = silver.groupBy("flight_month").agg(F.count("*").alias("so_dong"),
                                               F.round(F.avg("totalFare"), 2).alias("gia_tb")).orderBy("flight_month").toPandas()
by_month

# %%
# Tỉ lệ ghép thành công của từng nguồn ngoài (null = không ghép được)
match = silver.select(*[F.round(F.avg(F.col(c).isNotNull().cast("int")) * 100, 2).alias(c) for c in
                        ["origin_city", "dest_city", "days_to_holiday", "origin_temp_mean", "dest_temp_mean",
                         "jet_fuel_usd_gal", "cpi_all"]]).toPandas().T.rename(columns={0: "% dòng ghép được"})
match.to_csv(f"{EXPORTS}/03_silver_join_match_rate.csv")
match

# %%
silver.printSchema()
