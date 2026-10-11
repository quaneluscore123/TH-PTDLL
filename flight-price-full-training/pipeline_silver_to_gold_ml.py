"""Pipeline chạy toàn diện từ Bronze -> Silver -> Gold -> Huấn luyện Machine Learning
với tối đa lượng RAM (24 GB JVM Heap, 20 luồng i5-14500).

Đường dẫn dữ liệu:
    CSV: F:/PKA/Code/BD/itineraries.csv (31,09 GB)
    DATA_DIR: F:/flight-data

Chạy:
    python pipeline_silver_to_gold_ml.py
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_full import *
from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor


def step1_prepare_bronze(spark):
    """Bước 1: Nạp 31 GB CSV vào tầng Bronze Parquet zstd."""
    bronze_success = os.path.join(BRONZE, "_SUCCESS")
    if os.path.exists(bronze_success):
        print(f"[1/4] Tầng Bronze đã có sẵn tại: {BRONZE}")
        return

    print(f"\n=======================================================")
    print(f"[1/4] Đang nạp CSV 31 GB từ: {CSV_PATH}")
    print(f"=======================================================")
    t0 = time.time()
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
    raw = spark.read.csv(CSV_PATH, header=True, schema=schema, mode="PERMISSIVE")
    (raw.write.mode("overwrite")
        .option("compression", "zstd")
        .parquet(BRONZE))
    print(f"-> Ghi Bronze hoàn tất sau {(time.time() - t0) / 60:.1f} phút")


def step2_prepare_silver(spark):
    """Bước 2: Chuyển hóa Bronze thành Silver (82,1 triệu dòng, 57 cột, phân vùng theo flight_month)."""
    silver_success = os.path.join(SILVER, "_SUCCESS")
    if os.path.exists(silver_success):
        print(f"[2/4] Tầng Silver đã có sẵn tại: {SILVER}")
        return

    print(f"\n=======================================================")
    print(f"[2/4] Đang làm sạch và tạo đặc trưng tầng Silver (82 triệu dòng)...")
    print(f"=======================================================")
    t0 = time.time()
    bronze = spark.read.parquet(BRONZE)

    def seg(c, i=0):
        return F.split(F.col(c), r"\|\|").getItem(i)

    def iso_part(pattern):
        return F.coalesce(F.regexp_extract("travelDuration", pattern, 1).try_cast("int"), F.lit(0))

    cabin_rank = F.create_map(*[F.lit(x) for x in ["coach", 1, "premium coach", 2, "business", 3, "first", 4]])

    flights = (
        bronze
        .filter((F.col("totalFare") > 0) & F.col("searchDate").isNotNull() & F.col("flightDate").isNotNull())
        .withColumn("days_before", F.datediff("flightDate", "searchDate"))
        .filter(F.col("days_before") >= 0)
        .withColumn("booking_window", F.when(F.col("days_before") <= 7, "1. Rất gấp (1-7 ngày)")
                                       .when(F.col("days_before") <= 30, "2. Gấp (8-30 ngày)")
                                       .otherwise("3. Sớm (31-60 ngày)"))
        .withColumn("search_dow", F.dayofweek("searchDate"))
        .withColumn("flight_month", F.date_format("flightDate", "yyyy-MM"))
        .withColumn("month", F.month("flightDate"))
        .withColumn("week_of_year", F.weekofyear("flightDate"))
        .withColumn("day_of_week", F.dayofweek("flightDate"))
        .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
        .withColumn("dep_hour", F.substring(seg("segmentsDepartureTimeRaw"), 12, 2).try_cast("int"))
        .withColumn("route", F.concat_ws("-", "startingAirport", "destinationAirport"))
        .withColumn("duration_min", iso_part(r"P(\d+)D") * 1440 + iso_part(r"(\d+)H") * 60 + iso_part(r"(\d+)M"))
        .withColumn("num_stops", F.size(F.split("segmentsAirlineCode", r"\|\|")) - 1)
        .withColumn("airline_code", seg("segmentsAirlineCode"))
        .withColumn("airline_name", seg("segmentsAirlineName"))
        .withColumn("cabin_level", F.array_max(F.transform(F.split("segmentsCabinCode", r"\|\|"), lambda x: cabin_rank[x])))
        .withColumn("basic_economy", F.col("isBasicEconomy").cast("int"))
        .withColumn("refundable", F.col("isRefundable").cast("int"))
        .withColumn("non_stop", F.col("isNonStop").cast("int"))
        .select("legId", "searchDate", "flightDate", "flight_month", "startingAirport", "destinationAirport", "route",
                "airline_code", "airline_name", "fareBasisCode", "totalFare",
                "days_before", "booking_window", "search_dow", "month", "week_of_year", "day_of_week", "is_weekend",
                "dep_hour", "duration_min", "num_stops", "cabin_level", "basic_economy",
                "refundable", "non_stop", "seatsRemaining", "totalTravelDistance")
        .dropDuplicates(["legId", "searchDate", "fareBasisCode", "totalFare"])
    )

    # Ghép thông tin ngày lễ
    import holidays as holidays_lib
    us = holidays_lib.US(years=[2022])
    hol_df = pd.DataFrame({"flightDate": pd.to_datetime(list(us.keys())).date, "is_holiday": 1})
    hol_s = spark.createDataFrame(hol_df)

    silver = (
        flights
        .join(F.broadcast(hol_s), "flightDate", "left")
        .fillna({"is_holiday": 0})
    )

    silver.write.mode("overwrite").option("compression", "zstd").partitionBy("flight_month").parquet(SILVER)
    print(f"-> Tạo tầng Silver hoàn tất sau {(time.time() - t0) / 60:.1f} phút tại: {SILVER}")


def step3_prepare_gold(spark):
    """Bước 3: Tổng hợp tầng Gold và tính toán ma trận tương quan trên 82 triệu dòng."""
    print(f"\n=======================================================")
    print(f"[3/4] Đang phân tích và tổng hợp tầng Gold trên toàn bộ dữ liệu...")
    print(f"=======================================================")
    t0 = time.time()
    silver = spark.read.parquet(SILVER)

    # 1. Bảng Gold theo số ngày đặt trước
    print("-> Đang tính bảng tổng hợp theo booking_window và days_before...")
    gold_window = (
        silver.groupBy("booking_window")
        .agg(
            F.count("*").alias("n"),
            F.round(F.avg("totalFare"), 2).alias("avg_fare"),
            F.round(F.percentile_approx("totalFare", 0.5), 2).alias("median_fare")
        )
        .orderBy("booking_window")
    )
    gold_window.write.mode("overwrite").parquet(f"{GOLD}/by_booking_window")
    gold_window.toPandas().to_csv(f"{EXPORTS_FULL}/gold_by_booking_window.csv", index=False)

    gold_route = (
        silver.groupBy("route")
        .agg(
            F.count("*").alias("n"),
            F.round(F.avg("totalFare"), 2).alias("avg_fare"),
            F.round(F.percentile_approx("totalFare", 0.5), 2).alias("median_fare")
        )
    )
    gold_route.write.mode("overwrite").parquet(f"{GOLD}/by_route")

    # 2. Ma trận tương quan Pearson trên toàn bộ 82,1 triệu dòng
    print("-> Đang tính hệ số tương quan Pearson (Spark MLlib Correlation trên 82M dòng)...")
    num_cols = [
        "days_before", "duration_min", "num_stops", "non_stop",
        "cabin_level", "basic_economy", "refundable", "seatsRemaining",
        "totalTravelDistance", "dep_hour", "day_of_week", "search_dow",
        "is_weekend", "is_holiday"
    ]
    cols = ["totalFare", "log_fare"] + num_cols
    corr_df = silver.withColumn("log_fare", F.log("totalFare")).select(*[F.col(c).cast("double") for c in cols])
    vec = VectorAssembler(inputCols=cols, outputCol="v", handleInvalid="skip").transform(corr_df).select("v")
    pearson = Correlation.corr(vec, "v", "pearson").head()[0].toArray()

    corr_res = pd.DataFrame({
        "pearson_fare": pearson[0, 2:],
        "pearson_logfare": pearson[1, 2:],
    }, index=num_cols).round(4)
    corr_res["abs_max"] = corr_res.abs().max(axis=1)
    corr_res = corr_res.sort_values("abs_max", ascending=False)
    corr_res.to_csv(f"{EXPORTS_FULL}/correlation_with_fare.csv")
    print(corr_res)

    # 3. Tỷ số tương quan Eta cho biến định tính
    print("-> Đang tính tỷ số tương quan Eta cho biến phân loại...")
    overall = silver.agg(F.avg("totalFare").alias("m"), F.variance("totalFare").alias("v"), F.count("*").alias("n")).first()
    eta_dict = {}
    for c in ["route", "airline_code", "startingAirport", "destinationAirport", "booking_window"]:
        g = silver.groupBy(c).agg(F.count("*").alias("n"), F.avg("totalFare").alias("m")).toPandas()
        ss_between = (g.n * (g.m - overall.m) ** 2).sum()
        eta_dict[c] = round(float(np.sqrt(ss_between / (overall.v * (overall.n - 1)))), 4)
    eta_res = pd.Series(eta_dict, name="eta").sort_values(ascending=False)
    eta_res.to_csv(f"{EXPORTS_FULL}/eta_categorical.csv")
    print(eta_res)

    # 4. Lưu feature_selection.json
    selected_num = [f for f in corr_res.index if corr_res.loc[f, "abs_max"] >= 0.05 or f == "days_before"]
    selected_cat = [c for c in eta_res.index if eta_res[c] >= 0.10 and c != "booking_window"]
    selection = {"numeric": selected_num, "categorical": selected_cat}
    json.dump(selection, open(f"{GOLD}/feature_selection.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> Tầng Gold hoàn tất sau {(time.time() - t0) / 60:.1f} phút")


def step4_train_ml(spark):
    """Bước 4: Huấn luyện Machine Learning trên dữ liệu đầy đủ với 24 GB RAM."""
    print(f"\n=======================================================")
    print(f"[4/4] Bắt đầu huấn luyện Machine Learning với 24 GB RAM...")
    print(f"=======================================================")
    sel = json.load(open(f"{GOLD}/feature_selection.json", encoding="utf-8"))
    num_cols, cat_cols = sel["numeric"], sel["categorical"]

    TRAIN_MONTHS = ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"]
    TEST_MONTHS = ["2022-09", "2022-10"]

    silver = spark.read.parquet(SILVER).filter(F.col("flight_month").isin(TRAIN_MONTHS + TEST_MONTHS))
    selected_cols = list(dict.fromkeys(["totalFare", "flight_month", "booking_window"] + num_cols + cat_cols))
    data = (
        silver
        .select(*[c for c in selected_cols if c in silver.columns])
        .withColumn("label", F.log("totalFare"))
    )

    train_df = data.filter(F.col("flight_month").isin(TRAIN_MONTHS))
    test_df = data.filter(F.col("flight_month").isin(TEST_MONTHS))

    n_train = train_df.count()
    n_test = test_df.count()
    print(f"-> Tập Huấn luyện (Train T04-T08): {n_train:,} dòng")
    print(f"-> Tập Kiểm tra (Test T09-T10):    {n_test:,} dòng")
    print(f"-> Tổng cộng:                     {n_train + n_test:,} dòng")

    test_mean_fare = float(test_df.agg(F.avg("totalFare")).first()[0] or 300.0)
    MAX_BINS = 300

    def feature_stages(is_linear=False):
        imputed = [f"{c}_imp" for c in num_cols]
        stages = [Imputer(inputCols=num_cols, outputCols=imputed, strategy="median")]
        stages += [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
        if is_linear:
            stages += [OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols], outputCols=[f"{c}_ohe" for c in cat_cols])]
            stages += [
                VectorAssembler(inputCols=imputed, outputCol="num_vec"),
                StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True),
                VectorAssembler(inputCols=["num_scaled"] + [f"{c}_ohe" for c in cat_cols], outputCol="features")
            ]
        else:
            stages += [VectorAssembler(inputCols=imputed + [f"{c}_idx" for c in cat_cols], outputCol="features")]
        return stages

    def evaluate(pred_df, name, secs):
        p = pred_df.withColumn("pred_fare", F.exp("prediction"))
        m = p.select(
            F.sqrt(F.avg((F.col("pred_fare") - F.col("totalFare")) ** 2)).alias("RMSE"),
            F.avg(F.abs(F.col("pred_fare") - F.col("totalFare"))).alias("MAE"),
            (F.avg(F.abs(F.col("pred_fare") - F.col("totalFare")) / F.col("totalFare")) * 100).alias("MAPE_%"),
            (1 - F.sum((F.col("pred_fare") - F.col("totalFare")) ** 2)
                 / F.sum((F.col("totalFare") - F.lit(test_mean_fare)) ** 2)).alias("R2"),
        ).first().asDict()
        return {
            "model": name,
            "RMSE": round(float(m["RMSE"] or 0), 2),
            "MAE": round(float(m["MAE"] or 0), 2),
            "MAPE_%": round(float(m["MAPE_%"] or 0), 2),
            "R2": round(float(m["R2"] or 0), 4),
            "train_seconds": round(secs, 1)
        }

    results = []

    # 1. Baseline
    print("\n[ML 1/3] Đang tính Baseline (Trung vị Route x Booking Window)...")
    t0 = time.time()
    base_lut = train_df.groupBy("route", "booking_window").agg(F.percentile_approx("label", 0.5).alias("prediction"))
    fallback = float(train_df.agg(F.percentile_approx("label", 0.5)).first()[0] or 5.7)
    pred_base = test_df.join(base_lut, ["route", "booking_window"], "left").fillna({"prediction": fallback})
    res_base = evaluate(pred_base, "Baseline (Trung vị)", time.time() - t0)
    print(res_base)
    results.append(res_base)

    # 2. Linear Regression trên 100% dữ liệu
    print("\n[ML 2/3] Đang huấn luyện Linear Regression trên 100% dữ liệu...")
    t0 = time.time()
    pipe_lr = Pipeline(stages=feature_stages(is_linear=True) + [
        LinearRegression(featuresCol="features", labelCol="label", regParam=0.01, elasticNetParam=0.0, maxIter=50)
    ])
    model_lr = pipe_lr.fit(train_df)
    pred_lr = model_lr.transform(test_df)
    res_lr = evaluate(pred_lr, "Linear Regression (100% dữ liệu)", time.time() - t0)
    print(res_lr)
    results.append(res_lr)

    # 3. Random Forest (Tối ưu cho 24 GB RAM)
    print("\n[ML 3/3] Đang huấn luyện Random Forest Regressor (25 cây, tối ưu 24GB RAM)...")
    t0 = time.time()
    rf_algo = RandomForestRegressor(
        featuresCol="features", labelCol="label",
        numTrees=25, maxDepth=10, maxBins=MAX_BINS,
        subsamplingRate=0.5,
        featureSubsetStrategy="onethird",
        cacheNodeIds=True, checkpointInterval=5, seed=SEED
    )
    pipe_rf = Pipeline(stages=feature_stages(is_linear=False) + [rf_algo])
    model_rf = pipe_rf.fit(train_df)
    pred_rf = model_rf.transform(test_df)
    res_rf = evaluate(pred_rf, "Random Forest (Tối ưu 24GB RAM)", time.time() - t0)
    print(res_rf)
    results.append(res_rf)

    # Xuất kết quả
    res_df = pd.DataFrame(results).set_index("model")
    res_path = f"{EXPORTS_FULL}/model_comparison_full.csv"
    res_df.to_csv(res_path)
    print("\n=======================================================")
    print("   BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG MÔ HÌNH (24 GB RAM)  ")
    print("=======================================================")
    print(res_df.to_string())
    print(f"\nĐã lưu kết quả tại: {res_path}")

    # Lưu mô hình tốt nhất
    model_rf.write().overwrite().save(f"{MODELS_FULL}/best_model_full")
    print(f"Đã lưu mô hình tốt nhất tại: {MODELS_FULL}/best_model_full")


def main():
    total_t0 = time.time()
    print("==========================================================================")
    print("   KHỞI CHẠY PIPELINE BRONZE -> SILVER -> GOLD -> MACHINE LEARNING       ")
    print("   Dữ liệu gốc: F:/PKA/Code/BD/itineraries.csv (31,09 GB)                 ")
    print("   Tài nguyên: 24 GB JVM Heap | 20 luồng CPU Intel Core i5-14500         ")
    print("==========================================================================")

    spark = get_spark_32gb("pipeline-full-silver-gold-ml")

    step1_prepare_bronze(spark)
    step2_prepare_silver(spark)
    step3_prepare_gold(spark)
    step4_train_ml(spark)

    spark.stop()
    print(f"\n[HOÀN THÀNH] Toàn bộ pipeline đã chạy xong sau {(time.time() - total_t0) / 60:.1f} phút!")


if __name__ == "__main__":
    main()
