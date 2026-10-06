# %% [markdown]
# # 01 — Ingestion: nạp toàn bộ 82 triệu dòng vào tầng Bronze
#
# **Mốc pipeline:** Raw Data → Ingestion → **Bronze**
#
# - Nguồn: `itineraries.csv` (Expedia, Kaggle *dilwong/flightprices*), 31 GB, 82.138.753 dòng, 27 cột.
# - Đọc bằng Spark với **schema khai báo sẵn** (không để Spark quét 31 GB chỉ để đoán kiểu dữ liệu).
# - Ghi ra Parquet nén **zstd** trên ổ E — giữ nguyên toàn bộ dòng và cột (Bronze = dữ liệu thô, chưa làm sạch).

# %%
import sys, os, time
sys.path.insert(0, os.path.abspath(".."))
from common import *
from pyspark.sql import functions as F, types as T

spark = get_spark("01-bronze")
print("Spark", spark.version, "| CSV:", CSV_PATH, f"({os.path.getsize(CSV_PATH) / 1e9:.1f} GB)")

# %%
schema = T.StructType([
    T.StructField("legId", T.StringType()),
    T.StructField("searchDate", T.DateType()),
    T.StructField("flightDate", T.DateType()),
    T.StructField("startingAirport", T.StringType()),
    T.StructField("destinationAirport", T.StringType()),
    T.StructField("fareBasisCode", T.StringType()),
    T.StructField("travelDuration", T.StringType()),
    T.StructField("elapsedDays", T.IntegerType()),
    T.StructField("isBasicEconomy", T.BooleanType()),
    T.StructField("isRefundable", T.BooleanType()),
    T.StructField("isNonStop", T.BooleanType()),
    T.StructField("baseFare", T.DoubleType()),
    T.StructField("totalFare", T.DoubleType()),
    T.StructField("seatsRemaining", T.IntegerType()),
    T.StructField("totalTravelDistance", T.DoubleType()),
    T.StructField("segmentsDepartureTimeEpochSeconds", T.StringType()),
    T.StructField("segmentsDepartureTimeRaw", T.StringType()),
    T.StructField("segmentsArrivalTimeEpochSeconds", T.StringType()),
    T.StructField("segmentsArrivalTimeRaw", T.StringType()),
    T.StructField("segmentsArrivalAirportCode", T.StringType()),
    T.StructField("segmentsDepartureAirportCode", T.StringType()),
    T.StructField("segmentsAirlineName", T.StringType()),
    T.StructField("segmentsAirlineCode", T.StringType()),
    T.StructField("segmentsEquipmentDescription", T.StringType()),
    T.StructField("segmentsDurationInSeconds", T.StringType()),
    T.StructField("segmentsDistance", T.StringType()),
    T.StructField("segmentsCabinCode", T.StringType()),
])

# %%
t0 = time.time()
if not os.path.exists(f"{BRONZE}/_SUCCESS"):
    raw = spark.read.csv(CSV_PATH, header=True, schema=schema, mode="PERMISSIVE")
    (raw.write.mode("overwrite")
        .option("compression", "zstd")
        .parquet(BRONZE))
print(f"Ghi Bronze xong sau {(time.time() - t0) / 60:.1f} phút")

# %%
bronze = spark.read.parquet(BRONZE)
n = bronze.count()
size_gb = sum(os.path.getsize(os.path.join(BRONZE, f)) for f in os.listdir(BRONZE)) / 1e9
print(f"Số dòng Bronze: {n:,}")
print(f"Dung lượng Parquet: {size_gb:.2f} GB (CSV gốc: {os.path.getsize(CSV_PATH) / 1e9:.1f} GB)")
bronze.printSchema()

# %%
bronze.select("searchDate", "flightDate", "startingAirport", "destinationAirport",
              "totalFare", "seatsRemaining", "segmentsAirlineName").show(5, truncate=False)

# %% [markdown]
# ## Hồ sơ chất lượng dữ liệu thô (toàn bộ 82 triệu dòng)

# %%
profile = bronze.select(
    F.count("*").alias("so_dong"),
    *[F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in
      ["searchDate", "flightDate", "totalFare", "seatsRemaining", "totalTravelDistance", "segmentsAirlineName"]],
    F.sum((F.col("totalFare") <= 0).cast("int")).alias("gia_khong_hop_le"),
    F.min("searchDate").alias("searchDate_min"), F.max("searchDate").alias("searchDate_max"),
    F.min("flightDate").alias("flightDate_min"), F.max("flightDate").alias("flightDate_max"),
    F.countDistinct("legId").alias("so_legId"),
).toPandas().T.rename(columns={0: "giá trị"})
profile.to_csv(f"{EXPORTS}/01_bronze_profile.csv")
profile
