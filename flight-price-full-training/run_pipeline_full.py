"""Pipeline tự động đầu cuối (End-to-End Pipeline) cho máy 32 GB RAM (i5-14500):
1. Tự động kiểm tra tầng Silver: nếu chưa có, tự động nạp từ file CSV thô 31 GB và sinh tầng Silver 82,1 triệu dòng.
2. Huấn luyện mô hình Machine Learning ở quy mô bộ nhớ 32 GB RAM (20 luồng CPU).

Chạy:
    python run_pipeline_full.py                  # Chạy toàn bộ 100% dữ liệu
    python run_pipeline_full.py --fraction 0.5   # Huấn luyện trên 50% dữ liệu
    python run_pipeline_full.py --fraction 0.2   # Huấn luyện trên 20% dữ liệu
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import json
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_full import *
from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor


def ensure_silver_layer(spark):
    """Kiểm tra tầng Silver trên đĩa. Nếu chưa có, tự động nạp và tạo từ CSV gốc 31 GB."""
    silver_success = os.path.join(SILVER, "_SUCCESS")
    if os.path.exists(silver_success):
        print(f"[1/2] Tầng Silver đã sẵn sàng tại: {SILVER}")
        return

    print("\n[!] Chưa tìm thấy tầng Silver trên đĩa. Đang tự động tạo từ file CSV thô 31 GB...")
    bronze_success = os.path.join(DATA_DIR, "bronze", "itineraries", "_SUCCESS")
    BRONZE_DIR = f"{DATA_DIR}/bronze/itineraries"

    # Bước 1: Bronze Ingestion nếu chưa có
    if not os.path.exists(bronze_success):
        print(f"-> Đang nạp CSV 31 GB từ: {CSV_PATH}")
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
            .parquet(BRONZE_DIR))
        print(f"-> Ghi Bronze xong sau {(time.time() - t0) / 60:.1f} phút")

    # Bước 2: Silver ETL
    print("-> Đang làm sạch và tạo đặc trưng tầng Silver (82 triệu dòng)...")
    t0 = time.time()
    bronze = spark.read.parquet(BRONZE_DIR)

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
                "days_before", "booking_window", "search_dow", "day_of_week", "is_weekend",
                "dep_hour", "duration_min", "num_stops", "cabin_level", "basic_economy",
                "refundable", "non_stop", "seatsRemaining", "totalTravelDistance")
        .dropDuplicates(["legId", "searchDate", "fareBasisCode", "totalFare"])
    )

    # Ghép bảng ngày lễ nhanh qua thư viện holidays
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


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline huấn luyện Machine Learning quy mô 32GB RAM")
    parser.add_argument("--fraction", type=float, default=TRAIN_FRACTION,
                        help="Tỷ lệ dữ liệu huấn luyện: 1.0 (toàn bộ 82,1M dòng), 0.5 (50%%), 0.2 (20%%)")
    args = parser.parse_args()

    print("==========================================================================")
    print("   KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN MACHINE LEARNING - INTEL CORE i5-14500   ")
    print("   Bộ nhớ: 32 GB RAM (Driver: 22 GB Heap) | Số luồng: 20 threads         ")
    print("==========================================================================")

    spark = get_spark_32gb("pipeline-flight-32gb")

    # Đảm bảo Silver đã tồn tại
    ensure_silver_layer(spark)

    # Chuyển tiếp sang huấn luyện
    from train_full import load_feature_list, build_feature_stages, evaluate_predictions

    num_cols, cat_cols = load_feature_list()
    TRAIN_MONTHS = ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"]
    TEST_MONTHS = ["2022-09", "2022-10"]

    print("\n[2/2] Đang đọc dữ liệu tầng Silver để phân chia Train/Test...")
    base_df = spark.read.parquet(SILVER).filter(F.col("flight_month").isin(TRAIN_MONTHS + TEST_MONTHS))

    fraction = max(0.01, min(1.0, args.fraction))
    if fraction < 1.0:
        print(f"-> Đang lấy mẫu {fraction * 100:.1f}% từ tầng Silver...")
        base_df = base_df.sample(fraction=fraction, seed=SEED)

    selected_cols = ["totalFare", "flight_month", "days_before", "booking_window", "route"] + list(set(num_cols + cat_cols))
    data = (
        base_df
        .select(*[c for c in selected_cols if c in base_df.columns])
        .withColumn("label", F.log("totalFare"))
    )

    train_df = data.filter(F.col("flight_month").isin(TRAIN_MONTHS))
    test_df = data.filter(F.col("flight_month").isin(TEST_MONTHS))

    n_train = train_df.count()
    n_test = test_df.count()
    print(f"-> Tập Huấn luyện (Train T04-T08): {n_train:,} dòng")
    print(f"-> Tập Kiểm tra (Test T09-T10):    {n_test:,} dòng")

    test_mean_fare = float(test_df.agg(F.avg("totalFare")).first()[0] or 300.0)
    MAX_BINS = 300
    results = []

    # 1. Baseline
    print("\n--- 1/3. Huấn luyện Baseline (Trung vị Tuyến x Thời điểm đặt) ---")
    t0 = time.time()
    base_lut = train_df.groupBy("route", "booking_window").agg(F.percentile_approx("label", 0.5).alias("prediction"))
    fallback = float(train_df.agg(F.percentile_approx("label", 0.5)).first()[0] or 5.7)
    pred_base = test_df.join(base_lut, ["route", "booking_window"], "left").fillna({"prediction": fallback})
    res_base = evaluate_predictions(pred_base, "Baseline (Trung vị tuyến x booking window)", time.time() - t0, test_mean_fare)
    print(res_base)
    results.append(res_base)

    # 2. Linear Regression (100% data)
    print("\n--- 2/3. Huấn luyện Linear Regression (Hồi quy tuyến tính) ---")
    t0 = time.time()
    pipe_lr = Pipeline(stages=build_feature_stages(num_cols, cat_cols, is_linear=True) + [
        LinearRegression(featuresCol="features", labelCol="label", regParam=0.01, elasticNetParam=0.0, maxIter=50)
    ])
    model_lr = pipe_lr.fit(train_df)
    pred_lr = model_lr.transform(test_df)
    res_lr = evaluate_predictions(pred_lr, "Linear Regression", time.time() - t0, test_mean_fare)
    print(res_lr)
    results.append(res_lr)

    # 3. Random Forest (Memory-tuned cho 32GB RAM)
    print("\n--- 3/3. Huấn luyện Random Forest Regressor (25 cây) ---")
    t0 = time.time()
    subsample_rf = 0.5 if fraction >= 0.5 else 0.8
    rf_algo = RandomForestRegressor(
        featuresCol="features", labelCol="label",
        numTrees=25, maxDepth=10, maxBins=MAX_BINS,
        subsamplingRate=subsample_rf,
        featureSubsetStrategy="onethird",
        cacheNodeIds=True, checkpointInterval=5, seed=SEED
    )
    pipe_rf = Pipeline(stages=build_feature_stages(num_cols, cat_cols, is_linear=False) + [rf_algo])
    model_rf = pipe_rf.fit(train_df)
    pred_rf = model_rf.transform(test_df)
    res_rf = evaluate_predictions(pred_rf, "Random Forest (Tối ưu 32GB RAM)", time.time() - t0, test_mean_fare)
    print(res_rf)
    results.append(res_rf)

    # 4. LightGBM Regressor (Tối ưu RAM 32GB)
    print("\n--- 4/4. Huấn luyện LightGBM Regressor ---")
    t0 = time.time()
    feature_cols = [c for c in num_cols + cat_cols if c in train_df.columns]
    lgb_sample = train_df.sample(fraction=float(min(1.0, 3_000_000 / n_train)), seed=SEED) if n_train > 3_000_000 else train_df
    train_lgb_pdf = lgb_sample.select(*feature_cols, "label").toPandas()
    for c in cat_cols:
        if c in train_lgb_pdf.columns:
            train_lgb_pdf[c] = train_lgb_pdf[c].astype("category")

    lgb_algo = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    lgb_algo.fit(train_lgb_pdf[feature_cols], train_lgb_pdf["label"])
    lgb_secs = time.time() - t0

    test_eval_df = test_df if n_test <= 1_000_000 else test_df.sample(fraction=float(1_000_000 / n_test), seed=SEED)
    test_lgb_pdf = test_eval_df.select("totalFare", *feature_cols).toPandas()
    for c in cat_cols:
        if c in test_lgb_pdf.columns:
            test_lgb_pdf[c] = test_lgb_pdf[c].astype("category")
    test_lgb_pdf["prediction"] = lgb_algo.predict(test_lgb_pdf[feature_cols])
    pred_lgb_sdf = spark.createDataFrame(test_lgb_pdf[["totalFare", "prediction"]])
    res_lgb = evaluate_predictions(pred_lgb_sdf, "LightGBM (Tối ưu 32GB RAM)", lgb_secs, test_mean_fare)
    print(res_lgb)
    results.append(res_lgb)

    # Lưu mô hình LightGBM
    os.makedirs(MODELS_FULL, exist_ok=True)
    joblib.dump(lgb_algo, f"{MODELS_FULL}/best_model_lgb.joblib")
    lgb_algo.booster_.save_model(f"{MODELS_FULL}/best_model_lgb.txt")

    # Xuất kết quả
    res_df = pd.DataFrame(results).set_index("model")
    res_path = f"{EXPORTS_FULL}/model_comparison_full.csv"
    res_df.to_csv(res_path)
    print("\n==========================================================================")
    print("   KẾT QUẢ HUẤN LUYỆN TRÊN MÁY 32 GB RAM (i5-14500)                       ")
    print("==========================================================================")
    print(res_df.to_string())
    print(f"\nĐã lưu bảng kết quả tại: {res_path}")

    # Lưu mô hình tốt nhất
    best_name = res_df.drop(index=res_df.index[0])["MAE"].idxmin()
    if "LightGBM" in best_name:
        imp = pd.Series(lgb_algo.feature_importances_, index=feature_cols).sort_values(ascending=False)
        imp.to_csv(f"{EXPORTS_FULL}/feature_importance_full.csv", header=["importance"])
        print(f"Đã cập nhật feature importance từ LightGBM: {EXPORTS_FULL}/feature_importance_full.csv")

    model_rf.write().overwrite().save(f"{MODELS_FULL}/best_model_full")
    print(f"Đã lưu mô hình tốt nhất Spark (Random Forest) tại: {MODELS_FULL}/best_model_full")
    print(f"Mô hình đạt độ chính xác cao nhất tổng thể: {best_name}")

    spark.stop()
    print("\n[OK] Hoàn thành toàn bộ quy trình!")


if __name__ == "__main__":
    main()
