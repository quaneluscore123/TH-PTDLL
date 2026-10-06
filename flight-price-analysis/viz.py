"""Style biểu đồ dùng chung (matplotlib): bảng màu phân loại theo thứ tự cố định, nét mảnh, lưới nhạt."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Bảng màu phân loại — dùng theo đúng thứ tự, không xoay vòng
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#c23b3a"]   # xanh ↔ xám ↔ đỏ
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2, "lines.markersize": 5,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "patch.linewidth": 0,
    })


def note(ax, text):
    """Chú thích nhỏ màu xám dưới tiêu đề."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


setup()
