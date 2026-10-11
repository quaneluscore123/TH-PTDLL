# %% [markdown]
# # 05 — Machine Learning: huấn luyện tháng 4–8, kiểm tra trên tháng 9–10 (năm 2022)
#
# **Cách kiểm tra (out-of-time):** mô hình chỉ được học các chuyến bay có **ngày bay từ 04 → 08/2022**, rồi dự đoán
# giá vé cho các chuyến bay **tháng 09 và 10/2022** mà nó chưa từng thấy. Nếu sai số nhỏ, mô hình đủ tin cậy để
# dùng cho các tháng / năm tiếp theo (notebook 06).
#
# - Dữ liệu: mẫu ngẫu nhiên **5%** của Silver (≈ 4 triệu vé) — giới hạn RAM 16 GB của máy.
# - Chỉ số đầu vào: lấy từ kết quả **tương quan** ở notebook 04 (`feature_selection.json`).
# - So sánh **5 mô hình** (Spark MLlib): Baseline (trung vị theo tuyến × thời điểm đặt), Linear Regression,
#   Decision Tree, Random Forest, Gradient-Boosted Trees.
# - Thêm một thí nghiệm: **chỉ dùng dữ liệu chuyến bay** vs **có tích hợp nguồn ngoài** → giá trị của việc tích hợp.

# %%
import sys, os, json, time
sys.path.insert(0, os.path.abspath(".."))
from common import *
from viz import *
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor

spark = get_spark("05-ml")
sel = json.load(open(f"{GOLD}/feature_selection.json", encoding="utf-8"))
NUM, CAT = sel["numeric"], sel["categorical"]
print("Biến số:", NUM)
print("Biến phân loại:", CAT)

# %%
TRAIN_MONTHS = ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"]
TEST_MONTHS = ["2022-09", "2022-10"]

data = (spark.read.parquet(SILVER)
        .filter(F.col("flight_month").isin(TRAIN_MONTHS + TEST_MONTHS))
        .sample(fraction=ML_FRACTION, seed=SEED)
        .withColumn("label", F.log("totalFare"))
        .select("label", "totalFare", "flightDate", "flight_month", "days_before", "booking_window", "route",
                *set(NUM + CAT) - {"days_before", "route"}))
train = data.filter(F.col("flight_month").isin(TRAIN_MONTHS)).cache()
test = data.filter(F.col("flight_month").isin(TEST_MONTHS)).cache()
n_train, n_test = train.count(), test.count()
print(f"Train (bay 04–08/2022): {n_train:,} vé | Test (bay 09–10/2022): {n_test:,} vé")

# %% [markdown]
# ## Kiểm tra phạm vi giá trị (bổ sung cho bước chọn chỉ số bằng tương quan)
#
# Khi dự đoán **tương lai** (tháng 9–10), một chỉ số chỉ hữu ích nếu giá trị của nó ở tháng 9–10 đã từng xuất hiện
# trong dữ liệu huấn luyện. Các chỉ số chỉ "đánh dấu thời gian" như tuần trong năm hay CPI chung (tăng đều theo tháng)
# có giá trị tháng 9–10 **nằm ngoài** khoảng của tháng 4–8 → mô hình cây không ngoại suy được và sẽ học sai.
# Quy tắc: loại chỉ số nếu **> 50%** giá trị ở tập kiểm tra nằm ngoài [min, max] của tập huấn luyện.

# %%
rng = train.select(*[F.min(c).alias(f"{c}__min") for c in NUM], *[F.max(c).alias(f"{c}__max") for c in NUM]).first()
out_share = test.select(*[F.avg(((F.col(c) < rng[f"{c}__min"]) | (F.col(c) > rng[f"{c}__max"])).cast("double")).alias(c)
                          for c in NUM]).first().asDict()
range_check = pd.DataFrame({"min_train": [rng[f"{c}__min"] for c in NUM], "max_train": [rng[f"{c}__max"] for c in NUM],
                            "% test ngoài khoảng": [round((out_share[c] or 0) * 100, 1) for c in NUM]}, index=NUM)
range_check.to_csv(f"{EXPORTS}/05_range_check.csv")
OUT_OF_RANGE = [c for c in NUM if (out_share[c] or 0) > 0.5]
NUM = [c for c in NUM if c not in OUT_OF_RANGE]
print("Loại vì ngoài phạm vi huấn luyện:", OUT_OF_RANGE)
print("Còn lại", len(NUM), "biến số:", NUM)
range_check.sort_values("% test ngoài khoảng", ascending=False).head(10)

# %% [markdown]
# ## Pipeline đặc trưng
# - Biến phân loại → `StringIndexer` (cây quyết định) hoặc thêm `OneHotEncoder` (hồi quy tuyến tính).
# - Giá trị thiếu (khoảng cách bay, thời tiết...) → `Imputer` (trung vị).
# - Học trên **log(giá vé)** vì giá lệch phải mạnh; khi đánh giá đổi lại về USD.

# %%
def feature_stages(num, cat, linear=False):
    imputed = [f"{c}_imp" for c in num]
    stages = [Imputer(inputCols=num, outputCols=imputed, strategy="median")]
    stages += [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat]
    if linear:
        stages += [OneHotEncoder(inputCols=[f"{c}_idx" for c in cat], outputCols=[f"{c}_ohe" for c in cat])]
        stages += [VectorAssembler(inputCols=imputed, outputCol="num_vec"),
                   StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True),
                   VectorAssembler(inputCols=["num_scaled"] + [f"{c}_ohe" for c in cat], outputCol="features")]
    else:
        stages += [VectorAssembler(inputCols=imputed + [f"{c}_idx" for c in cat], outputCol="features")]
    return stages

def evaluate(pred, name, seconds):
    p = pred.withColumn("pred_fare", F.exp("prediction"))
    m = p.select(
        F.sqrt(F.avg((F.col("pred_fare") - F.col("totalFare")) ** 2)).alias("RMSE"),
        F.avg(F.abs(F.col("pred_fare") - F.col("totalFare"))).alias("MAE"),
        (F.avg(F.abs(F.col("pred_fare") - F.col("totalFare")) / F.col("totalFare")) * 100).alias("MAPE_%"),
        (1 - F.sum((F.col("pred_fare") - F.col("totalFare")) ** 2)
             / F.sum((F.col("totalFare") - F.lit(test_mean)) ** 2)).alias("R2"),
    ).first().asDict()
    return {"model": name, **{k: round(v, 4) for k, v in m.items()}, "train_seconds": round(seconds, 1)}

test_mean = test.agg(F.avg("totalFare")).first()[0]
MAX_BINS = 300   # ≥ số tuyến (≈ 235) để cây quyết định dùng được biến 'route'

# %% [markdown]
# ## Huấn luyện và so sánh các mô hình

# %%
results, preds, models = [], {}, {}

# 0) Baseline: trung vị log(giá) theo tuyến × nhóm thời điểm đặt, học trên tập train
t0 = time.time()
base = train.groupBy("route", "booking_window").agg(F.percentile_approx("label", 0.5).alias("prediction"))
fallback = train.agg(F.percentile_approx("label", 0.5)).first()[0]
pred_base = test.join(base, ["route", "booking_window"], "left").fillna({"prediction": fallback})
results.append(evaluate(pred_base, "Baseline (trung vị tuyến × thời điểm đặt)", time.time() - t0))
preds["Baseline"] = pred_base

algos = {
    "Linear Regression": (LinearRegression(featuresCol="features", labelCol="label", regParam=0.01, elasticNetParam=0.0, maxIter=50), True),
    "Decision Tree": (DecisionTreeRegressor(featuresCol="features", labelCol="label", maxDepth=12, maxBins=MAX_BINS, seed=SEED), False),
    # Tham số vừa với RAM 16 GB của máy (Spark local): rừng 30 cây sâu 10, GBT 60 vòng sâu 6
    "Random Forest": (RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=30, maxDepth=10,
                                            maxBins=MAX_BINS, subsamplingRate=0.4, featureSubsetStrategy="onethird",
                                            cacheNodeIds=True, checkpointInterval=10, seed=SEED), False),
    "Gradient-Boosted Trees": (GBTRegressor(featuresCol="features", labelCol="label", maxIter=60, maxDepth=6,
                                            stepSize=0.2, maxBins=MAX_BINS, subsamplingRate=0.6,
                                            cacheNodeIds=True, checkpointInterval=10, seed=SEED), False),
}
spark.sparkContext.setCheckpointDir(f"{DATA_DIR}/spark-tmp/checkpoints")
for name, (algo, linear) in algos.items():
    t0 = time.time()
    model = Pipeline(stages=feature_stages(NUM, CAT, linear) + [algo]).fit(train)
    secs = time.time() - t0
    pred = model.transform(test)            # không cache để tiết kiệm RAM
    results.append(evaluate(pred, name, secs))
    preds[name], models[name] = pred, model
    print(results[-1], flush=True)

# 5) LightGBM Regressor
print("\n--- Huấn luyện LightGBM Regressor ---")
t0 = time.time()
feature_cols = [c for c in NUM + CAT if c in train.columns]
train_pdf = train.select(*feature_cols, "label").toPandas()
test_eval = test.select("flightDate", "days_before", "booking_window", "totalFare", *feature_cols).toPandas()

for c in CAT:
    if c in train_pdf.columns:
        train_pdf[c] = train_pdf[c].astype("category")
        test_eval[c] = test_eval[c].astype("category")

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
lgb_algo.fit(train_pdf[feature_cols], train_pdf["label"])
secs_lgb = time.time() - t0

test_eval["prediction"] = lgb_algo.predict(test_eval[feature_cols])
pred_lgb = spark.createDataFrame(test_eval[["flightDate", "days_before", "booking_window", "totalFare", "prediction"]])
res_lgb = evaluate(pred_lgb, "LightGBM", secs_lgb)
results.append(res_lgb)
preds["LightGBM"] = pred_lgb
models["LightGBM"] = lgb_algo
print(results[-1], flush=True)

res = pd.DataFrame(results).set_index("model")
res.to_csv(f"{EXPORTS}/05_model_comparison.csv")
res

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
order = res.sort_values("MAE", ascending=False)
for ax, metric, fmt in zip(axes, ["MAE", "MAPE_%", "R2"], ["{:.1f} USD", "{:.1f}%", "{:.3f}"]):
    colors = [SERIES[1] if m == res[metric].idxmin() and metric != "R2" or m == res["R2"].idxmax() and metric == "R2"
              else SERIES[0] for m in order.index]
    ax.barh(order.index, order[metric], color=colors, height=0.6)
    for y, v in enumerate(order[metric]):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=INK_2)
    ax.set_title(metric); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    if ax is not axes[0]: ax.set_yticklabels([])
plt.suptitle("So sánh mô hình — dự đoán giá vé chuyến bay tháng 9–10/2022", x=0.01, ha="left", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{EXPORTS}/fig_05_model_comparison.png", dpi=150); plt.show()

# %% [markdown]
# ## Giá trị của việc tích hợp dữ liệu
# Huấn luyện lại mô hình tốt nhất **chỉ với các chỉ số có sẵn trong dữ liệu chuyến bay**, rồi so với bản có thêm
# ngày lễ, thời tiết, nhiên liệu, CPI, thông tin sân bay.

# %%
best_name = res.drop(index=res.index[0])["MAE"].idxmin()
EXTERNAL_COLS = {"is_holiday", "holiday_window", "days_to_holiday", "jet_fuel_usd_gal", "fuel_7d_change", "cpi_all",
                 "cpi_airfare", "inflation_yoy", "airfare_yoy", "cross_region", "lat_diff", "origin_region", "dest_region"}
EXTERNAL_COLS |= {c for c in NUM if c.startswith(("origin_", "dest_"))}
num_only = [c for c in NUM if c not in EXTERNAL_COLS]
cat_only = [c for c in CAT if c not in EXTERNAL_COLS]

if best_name == "LightGBM":
    t0 = time.time()
    lgb_only = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_only.fit(train_pdf[num_only + cat_only], train_pdf["label"])
    test_only_df = test_eval[["flightDate", "days_before", "booking_window", "totalFare"]].copy()
    test_only_df["prediction"] = lgb_only.predict(test_eval[num_only + cat_only])
    m_only = spark.createDataFrame(test_only_df)
    r_only = evaluate(m_only, f"{best_name} — chỉ dữ liệu chuyến bay", time.time() - t0)
else:
    algo, linear = algos[best_name]
    t0 = time.time()
    m_only = Pipeline(stages=feature_stages(num_only, cat_only, linear) + [algo.copy()]).fit(train)
    r_only = evaluate(m_only.transform(test), f"{best_name} — chỉ dữ liệu chuyến bay", time.time() - t0)

ablation = pd.DataFrame([r_only, {**results[list(res.index).index(best_name)], "model": f"{best_name} — có tích hợp nguồn ngoài"}]).set_index("model")
ablation.to_csv(f"{EXPORTS}/05_integration_ablation.csv")
print("Bỏ nguồn ngoài:", sorted((set(NUM) | set(CAT)) - set(num_only) - set(cat_only)))
ablation

# %% [markdown]
# ## Mức độ quan trọng của chỉ số (mô hình cây tốt nhất)

# %%
tree_name = best_name if best_name != "Linear Regression" else "LightGBM"
if tree_name == "LightGBM":
    feat_names = feature_cols
    imp = pd.Series(models["LightGBM"].feature_importances_, index=feat_names).sort_values()
else:
    fm = models[tree_name].stages[-1]
    feat_names = [f"{c}" for c in NUM] + CAT
    imp = pd.Series(fm.featureImportances.toArray(), index=feat_names).sort_values()

imp.to_csv(f"{EXPORTS}/05_feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(imp.index, imp.values, color=SERIES[0], height=0.65)
ax.set_title(f"Mức độ quan trọng của chỉ số — {tree_name}"); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(f"{EXPORTS}/fig_05_feature_importance.png", dpi=150); plt.show()

# %% [markdown]
# ## Dự đoán theo ngày: thực tế vs dự đoán cho tháng 9–10/2022

# %%
best = preds[best_name].withColumn("pred_fare", F.exp("prediction"))
by_day = (best.groupBy("flightDate").agg(F.avg("totalFare").alias("thuc_te"), F.avg("pred_fare").alias("du_doan"),
                                         F.count("*").alias("n")).orderBy("flightDate").toPandas())
by_db = (best.groupBy("days_before").agg(F.percentile_approx("totalFare", 0.5).alias("thuc_te"),
                                         F.percentile_approx("pred_fare", 0.5).alias("du_doan")).orderBy("days_before").toPandas())
by_day.to_csv(f"{EXPORTS}/05_test_by_flight_date.csv", index=False)
by_db.to_csv(f"{EXPORTS}/05_test_by_days_before.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
axes[0].plot(by_day.flightDate, by_day.thuc_te, color=SERIES[0], label="Thực tế")
axes[0].plot(by_day.flightDate, by_day.du_doan, color=SERIES[1], label=f"Dự đoán ({best_name})")
axes[0].set_title("Giá trung bình theo ngày bay — tháng 9–10/2022"); axes[0].set_ylabel("USD"); axes[0].legend()
axes[1].plot(by_db.days_before, by_db.thuc_te, color=SERIES[0], label="Thực tế")
axes[1].plot(by_db.days_before, by_db.du_doan, color=SERIES[1], label="Dự đoán")
axes[1].invert_xaxis(); axes[1].set_xlabel("Số ngày đặt trước"); axes[1].set_ylabel("Giá trung vị (USD)")
axes[1].set_title("Đường giá theo số ngày đặt trước — tháng 9–10/2022"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{EXPORTS}/fig_05_actual_vs_pred.png", dpi=150); plt.show()

# %%
# Sai số theo nhóm thời điểm đặt vé
err_window = (best.groupBy("booking_window")
              .agg(F.count("*").alias("n"), F.avg("totalFare").alias("gia_thuc_te_tb"), F.avg("pred_fare").alias("gia_du_doan_tb"),
                   (F.avg(F.abs(F.col("pred_fare") - F.col("totalFare")) / F.col("totalFare")) * 100).alias("MAPE_%"))
              .orderBy("booking_window").toPandas().round(2))
err_window.to_csv(f"{EXPORTS}/05_error_by_booking_window.csv", index=False)
err_window

# %% [markdown]
# ## Lưu mô hình tốt nhất (huấn luyện lại trên toàn bộ tháng 4–10) để dùng cho dự báo ở notebook 06

# %%
train.unpersist(); test.unpersist()
full = data.cache()
if best_name == "LightGBM":
    full_pdf = full.select(*feature_cols, "label").toPandas()
    for c in CAT:
        if c in full_pdf.columns:
            full_pdf[c] = full_pdf[c].astype("category")
    final_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1)
    final_lgb.fit(full_pdf[feature_cols], full_pdf["label"])
    os.makedirs(f"{DATA_DIR}/models", exist_ok=True)
    joblib.dump(final_lgb, f"{DATA_DIR}/models/best_model_lgb.joblib")
    final_lgb.booster_.save_model(f"{DATA_DIR}/models/best_model_lgb.txt")
else:
    algo, linear = algos[best_name]
    final = Pipeline(stages=feature_stages(NUM, CAT, linear) + [algo.copy()]).fit(full)
    final.write().overwrite().save(f"{DATA_DIR}/models/best_model")

json.dump({"best_model": best_name, "numeric": NUM, "categorical": CAT,
           "test_metrics": res.loc[best_name].to_dict()},
          open(f"{GOLD}/best_model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Đã lưu mô hình:", best_name)
