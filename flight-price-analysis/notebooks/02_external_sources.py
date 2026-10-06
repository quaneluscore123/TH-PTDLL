# %% [markdown]
# # 02 — Thu thập và chuẩn hóa các nguồn dữ liệu ngoài
#
# Bộ Expedia chỉ có thông tin chuyến bay của **năm 2022**. Để phân tích sâu hơn và dự báo cho 2023–2027,
# ta tích hợp thêm 7 nguồn:
#
# | # | Nguồn | Mức chi tiết | Dùng để |
# |---|---|---|---|
# | 1 | **OpenFlights** — danh mục sân bay | sân bay | thành phố, bang, tọa độ, múi giờ |
# | 2 | Thư viện **holidays** — ngày lễ liên bang Mỹ | ngày | `is_holiday`, số ngày tới ngày lễ gần nhất |
# | 3 | **Open-Meteo** — thời tiết lịch sử | sân bay × ngày | nhiệt độ, mưa, tuyết, gió |
# | 4 | **EIA** — giá nhiên liệu bay (Jet Fuel, U.S. Gulf Coast) | ngày, tháng | chi phí lớn nhất của hãng bay |
# | 5 | **BLS** — CPI vé máy bay & CPI chung (lạm phát) | tháng, 2005 → 08/2026 | xu hướng giá nhiều năm, lạm phát |
# | 6 | **DOT Consumer Airfare Report (Table 1a)** | tuyến × quý, 1993 → Q1/2024 | **giá vé thực tế 2023** để kiểm chứng |
# | 7 | **BTS Average Domestic Itinerary Fare** | sân bay × quý, 1993 → Q3/2024 | giá thực tế theo sân bay |
#
# Kết quả ghi vào `E:/flight-data/external/clean/` (CSV nhỏ, dùng cho Spark join và mô hình dự báo).

# %%
import sys, os, re, io, glob, time
sys.path.insert(0, os.path.abspath(".."))
from common import *
import pandas as pd, numpy as np, requests

CLEAN = f"{EXTERNAL}/clean"
os.makedirs(CLEAN, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36"}

# 16 sân bay có trong bộ Expedia (xem notebook 01 / xem_du_lieu.ipynb)
AIRPORTS = ["ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAD",
            "JFK", "LAX", "LGA", "MIA", "OAK", "ORD", "PHL", "SFO"]

# %% [markdown]
# ## 1. Sân bay — OpenFlights

# %%
ap_cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon",
           "alt", "tz_offset", "dst", "tzdb", "type", "source"]
ap = pd.read_csv("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
                 header=None, names=ap_cols, na_values="\\N")
airports = (ap[ap["iata"].isin(AIRPORTS)][["iata", "name", "city", "lat", "lon", "alt", "tzdb"]]
            .sort_values("iata").reset_index(drop=True))
# Bờ Đông / Trung tâm / Bờ Tây theo múi giờ
airports["region"] = airports["tzdb"].map({"America/New_York": "East", "America/Detroit": "East",
                                           "America/Chicago": "Central", "America/Denver": "Mountain",
                                           "America/Los_Angeles": "West"})
airports.to_csv(f"{CLEAN}/airports.csv", index=False)
airports

# %% [markdown]
# ## 2. Ngày lễ liên bang Mỹ — thư viện `holidays` (2022 → 2027)

# %%
import holidays as holidays_lib
us = holidays_lib.US(years=range(2022, 2028))
hol = (pd.DataFrame({"date": pd.to_datetime(list(us.keys())), "holiday_name": list(us.values())})
       .sort_values("date").reset_index(drop=True))
hol.to_csv(f"{CLEAN}/holidays.csv", index=False)
print(len(hol), "ngày lễ;", "trong khoảng bay của dữ liệu 2022:")
hol[(hol.date >= "2022-04-01") & (hol.date <= "2022-11-30")]

# %% [markdown]
# ## 3. Thời tiết lịch sử — Open-Meteo (16 sân bay, 01/04 → 30/11/2022, theo ngày)

# %%
frames = []
for _, a in airports.iterrows():
    for attempt in range(5):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
            "latitude": a.lat, "longitude": a.lon, "start_date": "2022-04-01", "end_date": "2022-11-30",
            "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum,snowfall_sum,wind_speed_10m_max",
            "timezone": a.tzdb})
        if r.status_code == 200:
            break
        time.sleep(10 * (attempt + 1))      # API giới hạn tần suất → chờ rồi thử lại
    r.raise_for_status()
    d = r.json()["daily"]
    frames.append(pd.DataFrame({
        "airport": a.iata, "date": pd.to_datetime(d["time"]),
        "temp_mean": d["temperature_2m_mean"], "temp_max": d["temperature_2m_max"],
        "precip_mm": d["precipitation_sum"], "snow_cm": d["snowfall_sum"], "wind_max": d["wind_speed_10m_max"]}))
    time.sleep(1)
weather = pd.concat(frames, ignore_index=True)
weather["bad_weather"] = ((weather.precip_mm >= 10) | (weather.snow_cm >= 1) | (weather.wind_max >= 40)).astype(int)
weather.to_csv(f"{CLEAN}/weather_daily.csv", index=False)
print(weather.shape)
weather.describe().round(1)

# %% [markdown]
# ## 4. Giá nhiên liệu bay — EIA (Kerosene-Type Jet Fuel, U.S. Gulf Coast, USD/gallon)

# %%
def eia_series(code, freq):
    path = f"{EXTERNAL}/eia_{code}_{freq}.xls"
    if not os.path.exists(path):
        r = requests.get(f"https://www.eia.gov/dnav/pet/hist_xls/{code}{freq}.xls", headers=UA, timeout=120)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    s = pd.read_excel(path, sheet_name="Data 1", skiprows=2, engine="xlrd")
    s.columns = ["date", "jet_fuel_usd_gal"]
    return s.dropna()

fuel_d = eia_series("EER_EPJK_PF4_RGC_DPG", "d")
fuel_m = eia_series("EER_EPJK_PF4_RGC_DPG", "m")
fuel_m["date"] = fuel_m["date"].dt.to_period("M").dt.to_timestamp()
fuel_d.to_csv(f"{CLEAN}/jet_fuel_daily.csv", index=False)
print("Daily:", fuel_d.date.min().date(), "→", fuel_d.date.max().date(),
      "| Monthly:", fuel_m.date.min().date(), "→", fuel_m.date.max().date())

# %% [markdown]
# ## 5. Lạm phát — BLS CPI (không điều chỉnh mùa vụ, 1982–84 = 100)
#
# - `CUUR0000SETG01`: CPI **vé máy bay** (Airline fares)
# - `CUUR0000SA0`: CPI **chung** (All items) → dùng để tính lạm phát

# %%
rows = []
for f in sorted(glob.glob(f"{EXTERNAL}/bls_*.json")):
    import json
    for s in json.load(open(f))["Results"]["series"]:
        for x in s["data"]:
            if x["period"].startswith("M") and x["period"] != "M13":
                value = float(x["value"]) if x["value"] not in ("-", "") else np.nan
                rows.append((s["seriesID"], f"{x['year']}-{x['period'][1:]}-01", value))
cpi = (pd.DataFrame(rows, columns=["series", "date", "value"]).drop_duplicates()
       .pivot(index="date", columns="series", values="value")
       .rename(columns={"CUUR0000SETG01": "cpi_airfare", "CUUR0000SA0": "cpi_all"}))
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi.sort_index()
missing = cpi[cpi.isna().any(axis=1)]
print("Tháng BLS không công bố số liệu (nội suy tuyến tính):", [d.strftime("%Y-%m") for d in missing.index])
cpi = cpi.interpolate()
cpi["inflation_yoy"] = cpi["cpi_all"].pct_change(12) * 100          # lạm phát so với cùng kỳ (%)
cpi["airfare_yoy"] = cpi["cpi_airfare"].pct_change(12) * 100

macro = cpi.join(fuel_m.set_index("date"), how="left").reset_index().rename(columns={"index": "date"})
macro.to_csv(f"{CLEAN}/macro_monthly.csv", index=False)
print("Macro theo tháng:", macro.date.min().date(), "→", macro.date.max().date(), macro.shape)
macro.tail(10).round(2)

# %% [markdown]
# ## 6. Giá vé thực tế theo tuyến — DOT Consumer Airfare Report, Table 1a
#
# Trang gốc `data.transportation.gov` chặn truy cập từ Việt Nam (HTTP 403), nên dùng bản sao trên Kaggle
# (*bhavikjikadara/us-airline-flight-routes-and-fares-1993-2024*). Đây là giá trung bình **một chiều** theo
# **cặp sân bay** (không phân biệt chiều đi/về), mỗi quý.

# %%
dot = pd.read_csv(glob.glob(f"{EXTERNAL}/bhavikjikadara*/*.csv")[0], low_memory=False)
dot = dot[dot.airport_1.isin(AIRPORTS) & dot.airport_2.isin(AIRPORTS)].copy()
dot["pair"] = [ "-".join(sorted(p)) for p in zip(dot.airport_1, dot.airport_2)]
# Một cặp sân bay có thể xuất hiện nhiều dòng trong cùng quý → lấy trung bình có trọng số theo số hành khách
dot = dot.rename(columns={"Year": "year"})
dot["w"] = dot["passengers"].clip(lower=1)
dot["fare_x_w"] = dot["fare"] * dot["w"]
dot_q = (dot.groupby(["pair", "year", "quarter"], as_index=False)
         .agg(fare_x_w=("fare_x_w", "sum"), w=("w", "sum"), passengers=("passengers", "sum"),
              nsmiles=("nsmiles", "mean"), fare_low=("fare_low", "mean")))
dot_q["fare"] = dot_q.pop("fare_x_w") / dot_q.pop("w")
dot_q.to_csv(f"{CLEAN}/dot_route_quarter.csv", index=False)
print("Số cặp sân bay:", dot_q.pair.nunique(), "| năm:", dot_q.year.min(), "→", dot_q.year.max())
dot_q.groupby("year").agg(so_tuyen=("pair", "nunique"), gia_tb=("fare", "mean")).tail(8).round(1)

# %% [markdown]
# ## 7. Giá vé trung bình theo sân bay — BTS (1993 → Q3/2024)
#
# File `.xls` thực chất là Excel XML 2003 và bị cắt cụt ở cuối, nên đọc bằng biểu thức chính quy.

# %%
def read_bts(path):
    text = open(path, encoding="utf-8-sig", errors="ignore").read()
    out = []
    for row in re.findall(r"<Row.*?</Row>", text, flags=re.S):
        cells = re.findall(r"<Data[^>]*>(.*?)</Data>", row, flags=re.S)
        if len(cells) >= 6 and cells[1] in AIRPORTS:
            out.append((cells[1], float(cells[5])))
    return out

bts_rows = []
for f in glob.glob(f"{EXTERNAL}/samithsachidanandan*/AverageFare_Q*_*.xls"):
    q, y = re.search(r"Q(\d)_(\d{4})", f).groups()
    bts_rows += [(code, int(y), int(q), fare) for code, fare in read_bts(f)]
bts = pd.DataFrame(bts_rows, columns=["airport", "year", "quarter", "avg_fare"]).sort_values(["airport", "year", "quarter"])
bts.to_csv(f"{CLEAN}/bts_airport_quarter.csv", index=False)
print(bts.shape, "| năm:", bts.year.min(), "→", bts.year.max())
bts.pivot_table(index="year", columns="airport", values="avg_fare").tail(6).round(0)

# %% [markdown]
# ## Tóm tắt các bảng đã chuẩn hóa

# %%
summary = pd.DataFrame([(os.path.basename(f), len(pd.read_csv(f)), round(os.path.getsize(f) / 1e3, 1))
                        for f in sorted(glob.glob(f"{CLEAN}/*.csv"))],
                       columns=["bảng", "số dòng", "KB"])
summary.to_csv(f"{EXPORTS}/02_external_sources_summary.csv", index=False)
summary
