"""Tạo dashboard HTML (một file, dữ liệu nhúng sẵn) từ các bảng kết quả trong E:/flight-data/exports.

    python dashboard/build_dashboard.py      → dashboard/index.html
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import EXPORTS, GOLD

HERE = os.path.dirname(os.path.abspath(__file__))


def csv(name, **kw):
    return pd.read_csv(f"{EXPORTS}/{name}", **kw)


def records(df, decimals=2):
    return json.loads(df.round(decimals).to_json(orient="records", force_ascii=False))


def build():
    D = {}
    kpi = csv("04_kpi.csv", index_col=0)["giá trị"]
    D["kpi"] = kpi.to_dict()

    D["booking"] = records(csv("gold_by_days_before.csv").query("days_before >= 1"))
    D["windows"] = records(csv("gold_by_booking_window.csv"))
    D["premAirline"] = records(csv("04_urgency_premium_airline.csv"))
    D["premRoute"] = records(csv("04_urgency_premium_route.csv"))
    for k, f in [("dow", "gold_by_day_of_week.csv"), ("hour", "gold_by_dep_hour.csv"), ("stops", "gold_by_stops.csv"),
                 ("holiday", "gold_by_days_to_holiday.csv"), ("month", "gold_by_flight_month.csv"),
                 ("airline", "gold_by_airline.csv"), ("seats", "gold_by_seats.csv")]:
        D[k] = records(csv(f).dropna())
    D["holiday"] = [r for r in D["holiday"] if abs(r["days_to_holiday"]) <= 10]

    corr = csv("04_correlation_with_fare.csv", index_col=0)
    D["corr"] = [{"feature": i, **r} for i, r in corr.round(4).to_dict(orient="index").items()]
    D["eta"] = csv("04_eta_categorical.csv", index_col=0)["eta"].round(4).to_dict()
    D["selection"] = json.load(open(f"{GOLD}/feature_selection.json", encoding="utf-8"))

    D["models"] = records(csv("05_model_comparison.csv"), 4)
    D["ablation"] = records(csv("05_integration_ablation.csv"), 4)
    D["testDay"] = records(csv("05_test_by_flight_date.csv"))
    D["testDb"] = records(csv("05_test_by_days_before.csv"))
    D["errWindow"] = records(csv("05_error_by_booking_window.csv"))
    imp = csv("05_feature_importance.csv", index_col=0)["importance"].sort_values(ascending=False)
    D["importance"] = imp.round(4).to_dict()
    D["bestModel"] = json.load(open(f"{GOLD}/best_model.json", encoding="utf-8"))

    D["macroCorr"] = records(csv("06_macro_correlation.csv"), 3)
    D["backtest"] = records(csv("06_backtest_2023_models.csv").rename(columns={"Unnamed: 0": "model"}))
    D["valid24"] = records(csv("06_validation_2024Q1.csv").rename(columns={"Unnamed: 0": "model"}))
    bt = csv("06_backtest_2023_detail.csv")
    D["backtestQ"] = records(bt.groupby("quarter").mean(numeric_only=True).reset_index())
    cpi = csv("06_forecast_cpi_airfare_monthly.csv")
    cpi["date"] = cpi["date"].str[:7]
    D["cpi"] = records(cpi)
    D["yearSummary"] = records(csv("06_forecast_summary_by_year.csv"))
    D["meta"] = json.load(open(f"{EXPORTS}/06_forecast_meta.json", encoding="utf-8"))

    rm = csv("06_forecast_route_month.csv")
    rm["month"] = rm["month"].str[:7]
    D["months"] = sorted(rm["month"].unique().tolist())
    D["routeFare"] = {r: g.sort_values("month")["expedia_fare"].round(1).tolist() for r, g in rm.groupby("route")}
    D["lastActual"] = rm.loc[~rm.is_forecast, "month"].max()
    cm = csv("06_booking_curve_multipliers.csv")
    D["curve"] = {r: g.sort_values("days_before")["mult"].round(4).tolist() for r, g in cm.groupby("route")}
    D["routeN"] = csv("gold_by_route.csv").set_index("route")["n"].to_dict()

    files = []
    for f in sorted(os.listdir(EXPORTS)):
        p = f"{EXPORTS}/{f}"
        if f.endswith(".csv"):
            files.append({"file": f, "rows": int(sum(1 for _ in open(p, encoding="utf-8")) - 1),
                          "kb": round(os.path.getsize(p) / 1024, 1)})
    D["files"] = files

    tpl = open(f"{HERE}/template.html", encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")))
    # artifact.html: phần thân trang (để đăng lên claude.ai); index.html: trang đầy đủ để mở trực tiếp trên máy
    open(f"{HERE}/artifact.html", "w", encoding="utf-8").write(html)
    full = ('<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n</head>\n<body>\n'
            + html + "\n</body>\n</html>\n")
    out = f"{HERE}/index.html"
    open(out, "w", encoding="utf-8").write(full)
    print("Đã ghi", out, f"({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
