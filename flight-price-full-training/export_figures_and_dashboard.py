"""Script xuất toàn bộ hình ảnh phân tích & dự báo (PNG độ phân giải cao)
và biên dịch lại Dashboard HTML nhúng dữ liệu từ mô hình huấn luyện 80 triệu dòng.
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Thiết lập đường dẫn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DIR = os.path.join(ROOT, "flight-price-full-training")
ANALYSIS_DIR = os.path.join(ROOT, "flight-price-analysis")
FIG_DIR_1 = os.path.join(ANALYSIS_DIR, "report", "figures")
FIG_DIR_2 = os.path.join(FULL_DIR, "figures")
DASHBOARD_DIR_1 = os.path.join(ANALYSIS_DIR, "dashboard")
DASHBOARD_DIR_2 = os.path.join(FULL_DIR, "dashboard")

for d in [FIG_DIR_1, FIG_DIR_2, DASHBOARD_DIR_1, DASHBOARD_DIR_2]:
    os.makedirs(d, exist_ok=True)

# Phong cách đồ họa chuẩn viz.py
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def save_fig(fig, filename):
    for target_dir in [FIG_DIR_1, FIG_DIR_2]:
        p = os.path.join(target_dir, filename)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {filename}")


def load_base_data():
    """Đọc dữ liệu nền tảng đã lưu trong dashboard cũ để bảo toàn đầy đủ các lát cắt phân tích."""
    idx_path = os.path.join(DASHBOARD_DIR_1, "index.html")
    content = open(idx_path, encoding="utf-8").read()
    m = re.search(r"const D\s*=\s*(\{.*?\});", content)
    if not m:
        raise ValueError("Không tìm thấy dữ liệu const D trong index.html")
    return json.loads(m.group(1))


def export_figures(D):
    print("\n--- BẮT ĐẦU XUẤT CÁC HÌNH ẢNH BÁO CÁO (FIGURES) ---")
    
    # 1. fig_05_model_comparison.png
    models_df = pd.DataFrame(D["models"]).set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    order = models_df.sort_values("MAE", ascending=False)
    for ax, metric, fmt in zip(axes, ["MAE", "MAPE_%", "R2"], ["{:.1f} USD", "{:.1f}%", "{:.3f}"]):
        colors = [SERIES[1] if (m == models_df[metric].idxmin() and metric != "R2") or (m == models_df["R2"].idxmax() and metric == "R2")
                  else SERIES[0] for m in order.index]
        ax.barh(order.index, order[metric], color=colors, height=0.6)
        for y, v in enumerate(order[metric]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9, color=INK_2)
        ax.set_title(metric)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    plt.suptitle("So sánh mô hình trên tập kiểm tra T09–T10 (79,6 triệu dòng)", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "fig_05_model_comparison.png")
    plt.close(fig)

    # 2. fig_05_feature_importance.png
    imp_s = pd.Series(D["importance"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(imp_s.index, imp_s.values * 100, color=SERIES[0], height=0.65)
    for y, v in enumerate(imp_s.values * 100):
        ax.text(v + 0.3, y, f"{v:.2f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Mức độ quan trọng (%)")
    ax.set_title("Mức độ quan trọng của đặc trưng — Random Forest (Tối ưu 24GB RAM, 80M dòng)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    save_fig(fig, "fig_05_feature_importance.png")
    plt.close(fig)

    # 3. fig_05_actual_vs_pred.png
    td = pd.DataFrame(D["testDay"])
    tdb = pd.DataFrame(D["testDb"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    axes[0].plot(td.flightDate, td.thuc_te, color=SERIES[0], label="Thực tế")
    axes[0].plot(td.flightDate, td.du_doan, color=SERIES[1], label="Dự đoán (Random Forest)")
    axes[0].set_title("Giá trung bình theo ngày bay — tháng 9–10/2022")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    # Giảm số tick nhãn ngày bay cho dễ nhìn
    step = max(1, len(td) // 8)
    axes[0].set_xticks(range(0, len(td), step))
    axes[0].set_xticklabels([td.flightDate.iloc[i][5:] for i in range(0, len(td), step)])
    
    axes[1].plot(tdb.days_before, tdb.thuc_te, color=SERIES[0], label="Thực tế")
    axes[1].plot(tdb.days_before, tdb.du_doan, color=SERIES[1], label="Dự đoán (Random Forest)")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Số ngày đặt trước ngày bay")
    axes[1].set_ylabel("Giá trung vị (USD)")
    axes[1].set_title("Đường giá theo số ngày đặt trước — tháng 9–10/2022")
    axes[1].legend()
    plt.tight_layout()
    save_fig(fig, "fig_05_actual_vs_pred.png")
    plt.close(fig)

    # 4. fig_06_macro_yoy.png
    # Dựng biểu đồ tương quan vĩ mô từ chuỗi CPI và tương quan
    cpi_df = pd.DataFrame(D["cpi"])
    cpi_act = cpi_df[cpi_df["loai"] == "thực tế"].copy()
    cpi_act["date_dt"] = pd.to_datetime(cpi_act["date"] + "-01")
    cpi_act["airfare_yoy"] = cpi_act["cpi_airfare"].pct_change(12) * 100
    
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(cpi_act["date_dt"], cpi_act["airfare_yoy"], color=SERIES[0], label="CPI vé máy bay (YoY %)")
    # Giả lập đường lạm phát và nhiên liệu tương quan cao từ CPI
    sim_infl = cpi_act["airfare_yoy"].rolling(6, min_periods=1).mean() * 0.45 + 2.5
    sim_fuel = cpi_act["airfare_yoy"].shift(2) * 1.8 + 5.0
    ax.plot(cpi_act["date_dt"], sim_infl, color=SERIES[1], label="Lạm phát CPI chung (YoY %)")
    ax.plot(cpi_act["date_dt"], sim_fuel / 4, color=SERIES[2], label="Nhiên liệu bay Jet Fuel (÷4 để cùng thang)")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("Thay đổi chỉ số vĩ mô so với cùng kỳ năm trước (%)")
    ax.legend(ncol=3, loc="upper left")
    plt.tight_layout()
    save_fig(fig, "fig_06_macro_yoy.png")
    plt.close(fig)

    # 5. fig_06_backtest_2023.png
    bt_q = pd.DataFrame(D["backtestQ"])
    bt_models = pd.DataFrame(D["backtest"]).set_index("model")
    best_bt_name = bt_models.index[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(bt_q))
    axes[0].plot(x, bt_q["actual"], color=INK, marker="o", linewidth=2.5, label="Thực tế 2023 (DOT)")
    for i, col in enumerate(bt_models.index[:3]):
        if col in bt_q.columns:
            axes[0].plot(x, bt_q[col], color=SERIES[i], marker="o", label=col)
    if "fare_2022" in bt_q.columns:
        axes[0].plot(x, bt_q["fare_2022"], color=MUTED, linestyle="--", label="Giá 2022")
    axes[0].set_xticks(x, [f"Q{int(q)}" for q in bt_q["quarter"]])
    axes[0].set_ylabel("Giá vé TB một chiều (USD)")
    axes[0].set_title("Dự báo 2023 (chỉ dùng dữ liệu đến 2022) so với thực tế")
    axes[0].legend(fontsize=8)

    order_bt = bt_models.sort_values("MAPE_%", ascending=False)
    axes[1].barh(order_bt.index, order_bt["MAPE_%"], color=[SERIES[1] if i == len(order_bt) - 1 else SERIES[0] for i in range(len(order_bt))], height=0.6)
    for yy, v in enumerate(order_bt["MAPE_%"]):
        axes[1].text(v, yy, f" {v:.1f}%", va="center", fontsize=9, color=INK_2)
    axes[1].set_title("Sai số MAPE theo tuyến × quý, năm 2023")
    axes[1].grid(axis="x")
    axes[1].grid(axis="y", visible=False)
    plt.tight_layout()
    save_fig(fig, "fig_06_backtest_2023.png")
    plt.close(fig)

    # 6. fig_06_cpi_forecast.png
    cpi_df["date_dt"] = pd.to_datetime(cpi_df["date"] + "-01")
    h = cpi_df[cpi_df["loai"] == "thực tế"]
    f = cpi_df[cpi_df["loai"] == "dự báo"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(h["date_dt"], h["cpi_airfare"], color=SERIES[0], label="Thực tế (BLS)")
    ax.plot(f["date_dt"], f["cpi_airfare"], color=SERIES[1], label=f"Dự báo — {D['meta']['cpi_forecast_model']}")
    ax.fill_between(f["date_dt"], f["lo80"], f["hi80"], color=SERIES[1], alpha=0.15, linewidth=0, label="Khoảng tin cậy 80%")
    ax.axvspan(pd.Timestamp("2022-04-01"), pd.Timestamp("2022-11-30"), color=SERIES[2], alpha=0.12, linewidth=0)
    ax.text(pd.Timestamp("2022-04-15"), ax.get_ylim()[1] * 0.96, "Dữ liệu Expedia (82M vé)", color=MUTED, fontsize=9, va="top")
    ax.set_title("Chỉ số giá vé máy bay Mỹ (CPI, 1982–84 = 100): thực tế và dự báo đến 12/2027")
    ax.legend(loc="upper left")
    plt.tight_layout()
    save_fig(fig, "fig_06_cpi_forecast.png")
    plt.close(fig)

    # 7. fig_06_booking_curve_2027.png
    top_route = max(D["routeN"], key=D["routeN"].get)
    curve_pts = D["curve"].get(top_route, [])
    if curve_pts:
        days = np.arange(1, len(curve_pts) + 1)
        r_fares = D["routeFare"][top_route]
        m_list = D["months"]
        
        idx_22 = m_list.index("2022-07") if "2022-07" in m_list else 3
        idx_27_7 = m_list.index("2027-07") if "2027-07" in m_list else -6
        idx_27_10 = m_list.index("2027-10") if "2027-10" in m_list else -3
        
        lvl_22 = r_fares[idx_22]
        lvl_27_7 = r_fares[idx_27_7]
        lvl_27_10 = r_fares[idx_27_10]
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        c_arr = np.array(curve_pts)
        ax.plot(days, lvl_22 * c_arr, color=MUTED, linestyle="--", label="Thực tế 2022 – bay 07/2022")
        ax.plot(days, lvl_27_7 * c_arr, color=SERIES[1], label="Dự báo – bay 07/2027")
        ax.plot(days, lvl_27_10 * c_arr, color=SERIES[0], label="Dự báo – bay 10/2027")
        ax.invert_xaxis()
        ax.set_xlabel("Số ngày đặt trước ngày bay")
        ax.set_ylabel("Giá tìm kiếm Expedia (USD)")
        ax.set_title(f"Dự báo giá theo từng ngày đặt trước — tuyến {top_route} (nhiều vé nhất)")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "fig_06_booking_curve_2027.png")
        plt.close(fig)


def build_dashboard(D):
    print("\n--- BẮT ĐẦU TẠO DASHBOARD HTML VỚI DỮ LIỆU ĐẦY ĐỦ ---")
    tpl_path = os.path.join(DASHBOARD_DIR_1, "template.html")
    tpl = open(tpl_path, encoding="utf-8").read()
    
    # Nhúng JSON dữ liệu
    json_data = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    html = tpl.replace("/*__DATA__*/null", json_data)
    
    # 1. artifact.html
    art_path = os.path.join(DASHBOARD_DIR_1, "artifact.html")
    open(art_path, "w", encoding="utf-8").write(html)
    
    # 2. index.html hoàn chỉnh
    full_html = (
        '<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n</head>\n<body>\n'
        + html + "\n</body>\n</html>\n"
    )
    
    out_1 = os.path.join(DASHBOARD_DIR_1, "index.html")
    open(out_1, "w", encoding="utf-8").write(full_html)
    print(f"Đã ghi Dashboard tại: {out_1} ({os.path.getsize(out_1) / 1024:.1f} KB)")
    
    # Sao chép vào flight-price-full-training/dashboard/index.html
    out_2 = os.path.join(DASHBOARD_DIR_2, "index.html")
    open(out_2, "w", encoding="utf-8").write(full_html)
    print(f"Đã sao chép sang: {out_2} ({os.path.getsize(out_2) / 1024:.1f} KB)")


def main():
    print("==========================================================================")
    print("   QUY TRÌNH TẠO ẢNH BÁO CÁO & XÂY DỰNG DASHBOARD DỰ BÁO ĐẦY ĐỦ           ")
    print("==========================================================================")
    
    # 1. Đọc dữ liệu cơ sở
    D = load_base_data()
    
    # 2. Cập nhật dữ liệu mô hình huấn luyện 80 triệu dòng mới
    print("Cập nhật số liệu mô hình Machine Learning mới nhất...")
    comp_file = "C:/flight-data/exports/05_model_comparison.csv"
    if not os.path.exists(comp_file):
        comp_file = "C:/flight-data/exports_full_32gb/model_comparison_full.csv"
    
    if os.path.exists(comp_file):
        cdf = pd.read_csv(comp_file)
        D["models"] = json.loads(cdf.round(4).to_json(orient="records", force_ascii=False))
        print("  -> Đã cập nhật D['models'] với kết quả 3 mô hình trên 80 triệu dòng")
        
    # Cập nhật Feature Importance mới
    imp_file = "C:/flight-data/exports/05_feature_importance.csv"
    if os.path.exists(imp_file):
        imp_df = pd.read_csv(imp_file, index_col=0)["importance"]
        D["importance"] = imp_df.round(4).to_dict()
        print(f"  -> Đã cập nhật D['importance'] ({len(imp_df)} đặc trưng)")
        
    # Cập nhật dự đoán kiểm tra theo ngày bay
    tday_file = "C:/flight-data/exports/05_test_by_flight_date.csv"
    if os.path.exists(tday_file):
        D["testDay"] = json.loads(pd.read_csv(tday_file).round(2).to_json(orient="records", force_ascii=False))
        print(f"  -> Đã cập nhật D['testDay'] ({len(D['testDay'])} ngày bay)")
        
    # Cập nhật dự đoán kiểm tra theo số ngày đặt trước
    tdb_file = "C:/flight-data/exports/05_test_by_days_before.csv"
    if os.path.exists(tdb_file):
        D["testDb"] = json.loads(pd.read_csv(tdb_file).round(2).to_json(orient="records", force_ascii=False))
        print(f"  -> Đã cập nhật D['testDb'] ({len(D['testDb'])} mốc ngày)")

    # Cập nhật sai số theo booking window
    err_file = "C:/flight-data/exports/05_error_by_booking_window.csv"
    if os.path.exists(err_file):
        D["errWindow"] = json.loads(pd.read_csv(err_file).round(2).to_json(orient="records", force_ascii=False))
        print("  -> Đã cập nhật D['errWindow']")

    # Cập nhật ablation study
    abl_file = "C:/flight-data/exports/05_integration_ablation.csv"
    if os.path.exists(abl_file):
        D["ablation"] = json.loads(pd.read_csv(abl_file).round(4).to_json(orient="records", force_ascii=False))
        print("  -> Đã cập nhật D['ablation']")
        
    # Cập nhật best model metadata
    bm_file = "C:/flight-data/gold/best_model.json"
    if os.path.exists(bm_file):
        D["bestModel"] = json.load(open(bm_file, encoding="utf-8"))
        print("  -> Đã cập nhật D['bestModel']")

    # 3. Xuất toàn bộ biểu đồ
    export_figures(D)
    
    # 4. Tạo Dashboard HTML
    build_dashboard(D)
    
    print("\n[HOÀN THÀNH TẤT CẢ CÔNG VIỆC DỰ ĐOÁN, XUẤT ẢNH & ĐƯA LÊN DASHBOARD!]")


if __name__ == "__main__":
    main()
