"""Kịch bản huấn luyện Machine Learning trên dữ liệu đầy đủ (lên đến 82,1 triệu dòng)
Tối ưu hóa cho cấu hình phần cứng: Intel Core i5-14500 + 32 GB RAM.

Cách chạy:
    python train_full.py                    # Huấn luyện 100% dữ liệu (mặc định)
    python train_full.py --fraction 0.50    # Huấn luyện trên 50% dữ liệu (~41 triệu dòng)
    python train_full.py --fraction 0.20    # Huấn luyện trên 20% dữ liệu (~16,4 triệu dòng)
"""
import sys
import os

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
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình giá vé máy bay trên dữ liệu lớn (32GB RAM)")
    parser.add_argument("--fraction", type=float, default=TRAIN_FRACTION,
                        help="Tỷ lệ lấy mẫu từ Silver: 1.0 (toàn bộ 82,1M dòng), 0.5 (50%%), 0.2 (20%%)")
    parser.add_argument("--skip-gbt", action="store_true", help="Bỏ qua GBT nếu chỉ muốn chạy nhanh Linear & Random Forest")
    parser.add_argument("--skip-lgb", action="store_true", help="Bỏ qua LightGBM")
    return parser.parse_args()


def load_feature_list():
    feature_file = f"{GOLD}/feature_selection.json"
    if os.path.exists(feature_file):
        sel = json.load(open(feature_file, encoding="utf-8"))
        return sel.get("numeric", []), sel.get("categorical", [])
    # Danh sách dự phòng chuẩn hóa
    num = [
        "totalTravelDistance", "duration_min", "num_stops", "non_stop",
        "cabin_level", "basic_economy", "refundable", "days_before",
        "lat_diff", "cross_region", "seatsRemaining", "dep_hour",
        "day_of_week", "search_dow", "is_weekend", "is_holiday",
        "days_to_holiday", "holiday_window", "jet_fuel_usd_gal", "fuel_7d_change"
    ]
    cat = ["route", "airline_code", "startingAirport", "destinationAirport"]
    return num, cat


def build_feature_stages(num_cols, cat_cols, is_linear=False):
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


def evaluate_predictions(pred_df, model_name, elapsed_seconds, test_mean_fare):
    p = pred_df.withColumn("pred_fare", F.exp("prediction"))
    metrics = p.select(
        F.sqrt(F.avg((F.col("pred_fare") - F.col("totalFare")) ** 2)).alias("RMSE"),
        F.avg(F.abs(F.col("pred_fare") - F.col("totalFare"))).alias("MAE"),
        (F.avg(F.abs(F.col("pred_fare") - F.col("totalFare")) / F.col("totalFare")) * 100).alias("MAPE_%"),
        (1 - F.sum((F.col("pred_fare") - F.col("totalFare")) ** 2)
             / F.sum((F.col("totalFare") - F.lit(test_mean_fare)) ** 2)).alias("R2"),
    ).first().asDict()
    return {
        "model": model_name,
        "RMSE": round(float(metrics["RMSE"] or 0), 2),
        "MAE": round(float(metrics["MAE"] or 0), 2),
        "MAPE_%": round(float(metrics["MAPE_%"] or 0), 2),
        "R2": round(float(metrics["R2"] or 0), 4),
        "train_seconds": round(elapsed_seconds, 1)
    }


def main():
    args = parse_args()
    fraction = max(0.01, min(1.0, args.fraction))
    print(f"=== BẮT ĐẦU HUẤN LUYỆN TRÊN MÁY 32GB RAM (i5-14500) ===")
    print(f"Tỷ lệ dữ liệu (Fraction): {fraction * 100:.1f}%")

    spark = get_spark_32gb("train-flight-32gb")
    num_cols, cat_cols = load_feature_list()

    TRAIN_MONTHS = ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"]
    TEST_MONTHS = ["2022-09", "2022-10"]

    print("Đọc dữ liệu từ tầng Silver Parquet:", SILVER)
    base_df = spark.read.parquet(SILVER).filter(F.col("flight_month").isin(TRAIN_MONTHS + TEST_MONTHS))

    if fraction < 1.0:
        base_df = base_df.sample(fraction=fraction, seed=SEED)

    selected_cols = list(dict.fromkeys(["totalFare", "flight_month", "booking_window"] + num_cols + cat_cols))
    data = (
        base_df
        .select(*[c for c in selected_cols if c in base_df.columns])
        .withColumn("label", F.log("totalFare"))
    )

    train_df = data.filter(F.col("flight_month").isin(TRAIN_MONTHS))
    test_df = data.filter(F.col("flight_month").isin(TEST_MONTHS))

    # Đếm số dòng thực tế
    print("Đang thống kê quy mô tập dữ liệu...")
    n_train = train_df.count()
    n_test = test_df.count()
    print(f"-> Tập Huấn luyện (Train T04-T08): {n_train:,} dòng")
    print(f"-> Tập Kiểm tra (Test T09-T10):    {n_test:,} dòng")
    print(f"-> Tổng cộng:                     {n_train + n_test:,} dòng")

    # Loại bỏ biến ngoài khoảng kiểm tra (> 50% out-of-range)
    print("Kiểm tra phân phối biên độ đặc trưng...")
    rng = train_df.select(*[F.min(c).alias(f"{c}__min") for c in num_cols], *[F.max(c).alias(f"{c}__max") for c in num_cols]).first()
    out_share = test_df.select(*[F.avg(((F.col(c) < rng[f"{c}__min"]) | (F.col(c) > rng[f"{c}__max"])).cast("double")).alias(c)
                                 for c in num_cols]).first().asDict()
    out_of_range = [c for c in num_cols if (out_share.get(c) or 0) > 0.5]
    if out_of_range:
        print(f"Loại biến trôi dạt theo thời gian: {out_of_range}")
        num_cols = [c for c in num_cols if c not in out_of_range]

    test_mean_fare = float(test_df.agg(F.avg("totalFare")).first()[0] or 300.0)
    MAX_BINS = 300

    results = []

    # 1. Baseline: Trung vị theo Route x Booking Window
    print("\n--- 1/4. Huấn luyện Baseline (Trung vị Route x Booking Window) ---")
    t0 = time.time()
    base_lut = train_df.groupBy("route", "booking_window").agg(F.percentile_approx("label", 0.5).alias("prediction"))
    fallback = float(train_df.agg(F.percentile_approx("label", 0.5)).first()[0] or 5.7)
    pred_base = test_df.join(base_lut, ["route", "booking_window"], "left").fillna({"prediction": fallback})
    res_base = evaluate_predictions(pred_base, "Baseline (Trung vị tuyến x thời điểm đặt)", time.time() - t0, test_mean_fare)
    print(res_base)
    results.append(res_base)

    # 2. Linear Regression trên 100% dữ liệu
    print("\n--- 2/4. Huấn luyện Linear Regression (Hồi quy tuyến tính) ---")
    t0 = time.time()
    pipe_lr = Pipeline(stages=build_feature_stages(num_cols, cat_cols, is_linear=True) + [
        LinearRegression(featuresCol="features", labelCol="label", regParam=0.01, elasticNetParam=0.0, maxIter=50)
    ])
    model_lr = pipe_lr.fit(train_df)
    pred_lr = model_lr.transform(test_df)
    res_lr = evaluate_predictions(pred_lr, "Linear Regression (100% dữ liệu)", time.time() - t0, test_mean_fare)
    print(res_lr)
    results.append(res_lr)

    # 3. Random Forest (Memory tuned cho 32GB RAM)
    print("\n--- 3/4. Huấn luyện Random Forest Regressor ---")
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
    res_rf = evaluate_predictions(pred_rf, "Random Forest (Tối ưu RAM 32GB)", time.time() - t0, test_mean_fare)
    print(res_rf)
    results.append(res_rf)

    # 4. Gradient-Boosted Trees (GBT)
    if not args.skip_gbt:
        print("\n--- 4/4. Huấn luyện Gradient-Boosted Trees (GBT) ---")
        t0 = time.time()
        subsample_gbt = 0.4 if fraction >= 0.5 else 0.6
        gbt_algo = GBTRegressor(
            featuresCol="features", labelCol="label",
            maxIter=40, maxDepth=6, stepSize=0.2, maxBins=MAX_BINS,
            subsamplingRate=subsample_gbt,
            cacheNodeIds=True, checkpointInterval=5, seed=SEED
        )
        pipe_gbt = Pipeline(stages=build_feature_stages(num_cols, cat_cols, is_linear=False) + [gbt_algo])
        model_gbt = pipe_gbt.fit(train_df)
        pred_gbt = model_gbt.transform(test_df)
        res_gbt = evaluate_predictions(pred_gbt, "Gradient-Boosted Trees (Tối ưu RAM 32GB)", time.time() - t0, test_mean_fare)
        print(res_gbt)
        results.append(res_gbt)

        # Trích xuất Feature Importance
        tree_model = model_gbt.stages[-1]
        feat_names = [f"{c}" for c in num_cols] + cat_cols
        imp = pd.Series(tree_model.featureImportances.toArray(), index=feat_names).sort_values(ascending=False)
        imp.to_csv(f"{EXPORTS_FULL}/feature_importance_full.csv", header=["importance"])

    # 5. LightGBM Regressor (Tối ưu RAM 32GB)
    lgb_algo = None
    feature_cols = [c for c in num_cols + cat_cols if c in train_df.columns]
    if not args.skip_lgb:
        total_steps = "5/5" if not args.skip_gbt else "4/4"
        print(f"\n--- {total_steps}. Huấn luyện LightGBM Regressor ---")
        t0 = time.time()

        # Đảm bảo an toàn bộ nhớ cho 32GB RAM khi JVM đã chiếm 24GB:
        # Lấy mẫu tối đa 3 triệu dòng cho pandas
        if n_train > 3_000_000:
            lgb_train_sample = train_df.sample(fraction=float(3_000_000 / n_train), seed=SEED)
        else:
            lgb_train_sample = train_df

        train_lgb_pdf = lgb_train_sample.select(*feature_cols, "label").toPandas()
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
        lgb_train_time = time.time() - t0

        # Đánh giá trên tập kiểm tra (mẫu 1M dòng nếu n_test lớn để tối ưu thời gian)
        test_eval_df = test_df if n_test <= 1_000_000 else test_df.sample(fraction=float(1_000_000 / n_test), seed=SEED)
        test_lgb_pdf = test_eval_df.select("totalFare", *feature_cols).toPandas()
        for c in cat_cols:
            if c in test_lgb_pdf.columns:
                test_lgb_pdf[c] = test_lgb_pdf[c].astype("category")

        test_lgb_pdf["prediction"] = lgb_algo.predict(test_lgb_pdf[feature_cols])
        pred_lgb_sdf = spark.createDataFrame(test_lgb_pdf[["totalFare", "prediction"]])
        res_lgb = evaluate_predictions(pred_lgb_sdf, "LightGBM (Tối ưu RAM 32GB)", lgb_train_time, test_mean_fare)
        print(res_lgb)
        results.append(res_lgb)

        # Lưu model LightGBM
        os.makedirs(MODELS_FULL, exist_ok=True)
        joblib.dump(lgb_algo, f"{MODELS_FULL}/best_model_lgb.joblib")
        lgb_algo.booster_.save_model(f"{MODELS_FULL}/best_model_lgb.txt")

    # Xuất bảng kết quả
    res_df = pd.DataFrame(results).set_index("model")
    res_path = f"{EXPORTS_FULL}/model_comparison_full.csv"
    res_df.to_csv(res_path)
    print("\n=== BẢNG SO SÁNH KẾT QUẢ HUẤN LUYỆN ===")
    print(res_df.to_string())
    print(f"\nĐã lưu kết quả tại: {res_path}")

    # Lưu mô hình tốt nhất
    best_name = res_df.drop(index=res_df.index[0])["MAE"].idxmin()
    if "LightGBM" in best_name and lgb_algo is not None:
        imp = pd.Series(lgb_algo.feature_importances_, index=feature_cols).sort_values(ascending=False)
        imp.to_csv(f"{EXPORTS_FULL}/feature_importance_full.csv", header=["importance"])
        print(f"Đã cập nhật feature importance từ LightGBM: {EXPORTS_FULL}/feature_importance_full.csv")

    spark_models = [m for m in res_df.index if m != res_df.index[0] and "LightGBM" not in m]
    best_spark_name = res_df.loc[spark_models]["MAE"].idxmin() if spark_models else "Random Forest"
    best_pipe = model_gbt if ("GBT" in best_spark_name and not args.skip_gbt) else (model_rf if "Random Forest" in best_spark_name else model_lr)
    best_pipe.write().overwrite().save(f"{MODELS_FULL}/best_model_full")
    print(f"Đã lưu mô hình tốt nhất Spark ({best_spark_name}) tại: {MODELS_FULL}/best_model_full")
    print(f"Mô hình đạt độ chính xác cao nhất tổng thể: {best_name}")

    spark.stop()


if __name__ == "__main__":
    main()
