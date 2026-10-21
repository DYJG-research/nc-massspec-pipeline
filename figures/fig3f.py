# ============================================================
# Fig. 2f
# Formula-Lattice validation
#
# Broken-axis bar chart
#
# Show:
#   - Before Formula-Lattice
#   - After Formula-Lattice
#   - Reduction marker overlapping the right edge of grey bar
#   - Marker shape like "工"
#   - Percentage only, with triangle marker
#
# All label positions are manually adjustable.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# 1. Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,

    "axes.linewidth": 0.8,
    "axes.labelsize": 7.2,

    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 150,
    "savefig.dpi": 600,
})


# ============================================================
# 2. Figure size
# ============================================================

FIG_WIDTH = 3.45
FIG_HEIGHT = 2.65


# ============================================================
# 3. Data
# ============================================================

settings = [
    "Top-1",
    "Top-3",
    "Top-5",
    "Top-10",
    "Ref.",
]

x = np.arange(len(settings))

before_counts = np.array([
    3915,
    2750,
    2495,
    2287,
    3272
])

after_counts = np.array([
    403,
    67,
    29,
    13,
    134
])

TOTAL_N = 17556

delta_counts = before_counts - after_counts
delta_rates = delta_counts / TOTAL_N * 100


# ============================================================
# 4. Colors
# ============================================================

COLOR_BEFORE = "#BDBDBD"
COLOR_AFTER = "#6C8EBF"
COLOR_DELTA = "#D55E00"
GRID_COLOR = "#D9D9D9"


# ============================================================
# 5. Broken-axis range
# ============================================================

TOP_YMIN = 1800
TOP_YMAX = 4300

BOTTOM_YMIN = 0
BOTTOM_YMAX = 550


# ============================================================
# 6. Font size configuration
# ============================================================

BAR_LABEL_FONTSIZE = 6.3
PCT_LABEL_FONTSIZE = 6.1
LEGEND_FONTSIZE = 6.0


# ============================================================
# 7. Legend position
# ============================================================

LEGEND_BBOX = (
    0.5,
    0.85
)


# ============================================================
# 8. Create broken axes
# ============================================================

fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.5, 1.0],
        "hspace": 0.05
    }
)


# ============================================================
# 9. Bars
# ============================================================

bar_width = 0.32

for ax in [ax_top, ax_bot]:
    ax.bar(
        x - bar_width / 2,
        before_counts,
        width=bar_width,
        color=COLOR_BEFORE,
        edgecolor="none",
        zorder=2
    )

    ax.bar(
        x + bar_width / 2,
        after_counts,
        width=bar_width,
        color=COLOR_AFTER,
        edgecolor="none",
        zorder=3
    )


# ============================================================
# 10. Manual coordinate configuration
# ============================================================
#
# 你后面如果要手调位置，主要改这里
#
# 每个 label 配置格式:
# {
#     "x": ...,
#     "y": ...,
#     "dx": ...,
#     "dy": ...,
#     "ha": ...,
#     "va": ...
# }
#
# x, y 是锚点坐标
# dx, dy 是相对偏移, 单位是 points
# ============================================================

# ------------------------------------------------------------
# Grey bar labels
# ------------------------------------------------------------
before_label_cfg = [
    {"x": x[0] - bar_width / 2, "y": before_counts[0], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},  # Top-1, 3915
    {"x": x[1] - bar_width / 2, "y": before_counts[1], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},  # Top-3, 2750
    {"x": x[2] - bar_width / 2, "y": before_counts[2], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},  # Top-5, 2495
    {"x": x[3] - bar_width / 2, "y": before_counts[3], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},  # Top-10, 2287
    {"x": x[4] - bar_width / 2, "y": before_counts[4], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},  # Oracle, 3272
]

# ------------------------------------------------------------
# Blue bar labels
# ------------------------------------------------------------
after_label_cfg = [
    {"x": x[0] + bar_width / 2, "y": after_counts[0], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},   # Top-1, 403
    {"x": x[1] + bar_width / 2, "y": after_counts[1], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},   # Top-3, 67
    {"x": x[2] + bar_width / 2, "y": after_counts[2], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},   # Top-5, 29
    {"x": x[3] + bar_width / 2, "y": after_counts[3], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},   # Top-10, 13
    {"x": x[4] + bar_width / 2, "y": after_counts[4], "dx": 0, "dy": 1, "ha": "center", "va": "bottom"},   # Oracle, 134
]

# ------------------------------------------------------------
# Reduction marker configuration
#
# x:
#   直接与灰色柱子的右边界重合
#   对于灰色柱，其右边界正好就是 x[i]
#
# y_top / y_bottom:
#   差值线顶部和底部锚点
#
# cap_half_width_fig:
#   “工”字顶部/底部横线向左右各伸出的长度
#   单位是 figure fraction
# ------------------------------------------------------------
reduction_marker_cfg = [
    {"x": x[0], "y_top": before_counts[0], "y_bottom": after_counts[0], "cap_half_width_fig": 0.016},  # Top-1
    {"x": x[1], "y_top": before_counts[1], "y_bottom": after_counts[1], "cap_half_width_fig": 0.016},  # Top-3
    {"x": x[2], "y_top": before_counts[2], "y_bottom": after_counts[2], "cap_half_width_fig": 0.016},  # Top-5
    {"x": x[3], "y_top": before_counts[3], "y_bottom": after_counts[3], "cap_half_width_fig": 0.016},  # Top-10
    {"x": x[4], "y_top": before_counts[4], "y_bottom": after_counts[4], "cap_half_width_fig": 0.016},  # Oracle
]

# ------------------------------------------------------------
# Percentage label configuration
# ------------------------------------------------------------
pct_label_cfg = [
    {"x": reduction_marker_cfg[0]["x"], "y": reduction_marker_cfg[0]["y_top"], "dx": 0, "dy": -50, "ha": "left", "va": "bottom"},  # ▲20.00%
    {"x": reduction_marker_cfg[1]["x"], "y": reduction_marker_cfg[1]["y_top"], "dx": 0, "dy": -25, "ha": "left", "va": "bottom"},  # ▲15.28%
    {"x": reduction_marker_cfg[2]["x"], "y": reduction_marker_cfg[2]["y_top"], "dx": 0, "dy": -20, "ha": "left", "va": "bottom"},  # ▲14.05%
    {"x": reduction_marker_cfg[3]["x"], "y": reduction_marker_cfg[3]["y_top"], "dx": 0, "dy": -10, "ha": "left", "va": "bottom"},  # ▲12.95%
    {"x": reduction_marker_cfg[4]["x"], "y": reduction_marker_cfg[4]["y_top"], "dx": 0, "dy": -35, "ha": "left", "va": "bottom"},  # ▲17.87%
]


# ============================================================
# 11. Broken-axis settings
# ============================================================

ax_top.set_ylim(TOP_YMIN, TOP_YMAX)
ax_bot.set_ylim(BOTTOM_YMIN, BOTTOM_YMAX)

ax_top.set_yticks([2000, 3000, 4000])
ax_bot.set_yticks([0, 200, 400])

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(settings)

ax_bot.set_xlim(-0.45, len(settings) - 0.55)


# ============================================================
# 12. Grid
# ============================================================

for ax in [ax_top, ax_bot]:
    ax.grid(
        axis="y",
        linestyle=(0, (3, 3)),
        linewidth=0.6,
        color=GRID_COLOR
    )
    ax.set_axisbelow(True)


# ============================================================
# 13. Axis style
# ============================================================

for ax in [ax_top, ax_bot]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax_top.spines["bottom"].set_visible(False)

ax_top.tick_params(
    axis="x",
    bottom=False,
    labelbottom=False
)


# ============================================================
# 14. Break marks
# ============================================================

d = 0.012

ax_top.plot(
    (-d, +d),
    (-d, +d),
    transform=ax_top.transAxes,
    color="black",
    clip_on=False,
    linewidth=0.8
)

ax_bot.plot(
    (-d, +d),
    (1 - d, 1 + d),
    transform=ax_bot.transAxes,
    color="black",
    clip_on=False,
    linewidth=0.8
)


# ============================================================
# 15. Layout
# ============================================================

plt.subplots_adjust(
    left=0.18,
    right=0.93,
    top=0.78,
    bottom=0.18
)


# ============================================================
# 16. Helper functions for figure-coordinate drawing
# ============================================================

def data_x_to_fig(ax, xdata):
    xmin, xmax = ax.get_xlim()
    pos = ax.get_position()
    return pos.x0 + (xdata - xmin) / (xmax - xmin) * pos.width


def data_y_to_fig(ax, ydata):
    ymin, ymax = ax.get_ylim()
    pos = ax.get_position()
    return pos.y0 + (ydata - ymin) / (ymax - ymin) * pos.height


# ============================================================
# 17. Draw bar value labels
# ============================================================

for cfg, val in zip(before_label_cfg, before_counts):
    ax_top.annotate(
        f"{val:,}",
        xy=(cfg["x"], cfg["y"]),
        xytext=(cfg["dx"], cfg["dy"]),
        textcoords="offset points",
        ha=cfg["ha"],
        va=cfg["va"],
        fontsize=BAR_LABEL_FONTSIZE,
        color="#707070",
        fontweight="bold",
        zorder=10,
    )

for cfg, val in zip(after_label_cfg, after_counts):
    ax_bot.annotate(
        f"{val:,}",
        xy=(cfg["x"], cfg["y"]),
        xytext=(cfg["dx"], cfg["dy"]),
        textcoords="offset points",
        ha=cfg["ha"],
        va=cfg["va"],
        fontsize=BAR_LABEL_FONTSIZE,
        color=COLOR_AFTER,
        fontweight="bold",
        zorder=10,
    )


# ============================================================
# 18. Draw single-piece reduction markers, "工" shape
# ============================================================

for cfg in reduction_marker_cfg:
    x_fig = data_x_to_fig(ax_bot, cfg["x"])
    y_top_fig = data_y_to_fig(ax_top, cfg["y_top"])
    y_bottom_fig = data_y_to_fig(ax_bot, cfg["y_bottom"])
    cap = cfg["cap_half_width_fig"]

    # vertical line
    fig.lines.append(
        Line2D(
            [x_fig, x_fig],
            [y_bottom_fig, y_top_fig],
            transform=fig.transFigure,
            color=COLOR_DELTA,
            linewidth=1.1,
            zorder=50,
        )
    )

    # top horizontal line
    fig.lines.append(
        Line2D(
            [x_fig - cap, x_fig + cap],
            [y_top_fig, y_top_fig],
            transform=fig.transFigure,
            color=COLOR_DELTA,
            linewidth=1.1,
            zorder=50,
        )
    )

    # bottom horizontal line
    fig.lines.append(
        Line2D(
            [x_fig - cap, x_fig + cap],
            [y_bottom_fig, y_bottom_fig],
            transform=fig.transFigure,
            color=COLOR_DELTA,
            linewidth=1.1,
            zorder=50,
        )
    )


# ============================================================
# 19. Draw percentage labels
# ============================================================

for cfg, rate in zip(pct_label_cfg, delta_rates):
    ax_top.annotate(
        f"▲{rate:.2f}%",
        xy=(cfg["x"], cfg["y"]),
        xytext=(cfg["dx"], cfg["dy"]),
        textcoords="offset points",
        ha=cfg["ha"],
        va=cfg["va"],
        fontsize=PCT_LABEL_FONTSIZE,
        color=COLOR_DELTA,
        fontweight="bold",
        zorder=60,
    )


# ============================================================
# 20. Centered left ylabel
# ============================================================

pos_top = ax_top.get_position()
pos_bot = ax_bot.get_position()
center_y = (pos_top.y1 + pos_bot.y0) / 2

fig.text(
    0.05,
    center_y,
    "Empty predictions",
    rotation=90,
    ha="center",
    va="center",
    fontsize=7.2
)


# ============================================================
# 21. Legend
# ============================================================

legend_handles = [
    Patch(
        facecolor=COLOR_BEFORE,
        edgecolor="none",
        label="Before Formula-Lattice"
    ),
    Patch(
        facecolor=COLOR_AFTER,
        edgecolor="none",
        label="After Formula-Lattice"
    ),
    Line2D(
        [0], [0],
        color=COLOR_DELTA,
        linewidth=1.2,
        label="Reduction rate"
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=LEGEND_BBOX,
    ncol=3,
    frameon=False,
    fontsize=LEGEND_FONTSIZE,
    columnspacing=1.0,
    handlelength=1.8
)


# ============================================================
# 22. Save
# ============================================================

out_dir = Path("./figures")
out_dir.mkdir(
    parents=True,
    exist_ok=True
)

pdf_path = out_dir / "fig2f_formula_lattice_reduction_gong_shape.pdf"
svg_path = out_dir / "fig2f_formula_lattice_reduction_gong_shape.svg"
png_path = out_dir / "fig2f_formula_lattice_reduction_gong_shape.png"

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    svg_path,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    png_path,
    bbox_inches="tight",
    facecolor="white",
    dpi=600
)

plt.show()

print("Saved:")
print(pdf_path)
print(svg_path)
print(png_path)