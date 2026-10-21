# ============================================================
# Fig. 2b
# Ablation analysis
#
# Three compact slope charts
#
# Panels:
#   Accuracy
#   MCES
#   ECFP4
#
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


# ============================================================
# 1. Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,

    "axes.linewidth": 0.8,
    "axes.labelsize": 7.4,

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
#
# 想让整个图左右更短，就主要改这两个：
#
# FIG_WIDTH  越小，整张图越窄
# FIG_HEIGHT 越小，整张图越矮
# ============================================================

FIG_WIDTH = 4.5
FIG_HEIGHT = 2.05


# ============================================================
# 3. Data
# ============================================================

methods = [
    "w/o FL + w/o XGB + w/o LLM",
    "w/o XGB + w/o LLM",
    "Our Model",
]

colors = [
    "#9E9E9E",   # baseline 1, gray
    "#6C8EBF",   # baseline 2, blue-gray
    "#D55E00",   # our model, orange-red
]

# Accuracy
accuracy = np.array([
    [12.63, 19.33],
    [13.03, 25.68],
    [22.37, 28.59],
])

# MCES (lower is better)
mces = np.array([
    [27.83, 24.51],
    [16.83, 11.24],
    [15.07, 11.03],
])

# ECFP4
ecfp4 = np.array([
    [0.32, 0.39],
    [0.35, 0.50],
    [0.44, 0.52],
])


# ============================================================
# 4. Manual label position config
# ============================================================
#
# 每个 tuple 格式:
#   (dx, dy, ha, va)
#
# 含义:
#   dx, dy: 相对数据点的偏移, 单位是 points
#   ha:     horizontal alignment
#   va:     vertical alignment
#
# 你之后只改这些值即可精调标签位置
# ============================================================

# ------------------------------------------------------------
# Accuracy panel
# method index:
#   0 -> w/o FL + w/o XGB + w/o LLM
#   1 -> w/o XGB + w/o LLM
#   2 -> Our Model
#
# x index:
#   0 -> Top-1
#   1 -> Top-10
# ------------------------------------------------------------
accuracy_label_cfg = {
    0: [( 7,  -6, "right", "center"), ( 8,  -8, "right", "center")],
    1: [( -2,  6, "center", "bottom"), ( 0,  -14, "center", "bottom")],
    2: [( -6,  3, "left",   "bottom"), ( -6,  2, "left",   "bottom")],
}

# ------------------------------------------------------------
# MCES panel
# ------------------------------------------------------------
mces_label_cfg = {
    0: [( 9,  5, "right", "center"), (10,  3, "right", "bottom")],
    1: [( 2,  2, "center", "bottom"), ( 0,  3, "center", "bottom")],
    2: [( -5, -8, "left",  "center"), ( -10,  -6, "left",   "center")],
}

# ------------------------------------------------------------
# ECFP4 panel
# ------------------------------------------------------------
ecfp4_label_cfg = {
    0: [( 7,  -6, "right", "center"), ( 8,  -6, "right", "center")],
    1: [( -2,  2, "center", "bottom"), ( 2,  -10, "center", "bottom")],
    2: [( -7,  2, "left",   "bottom"), ( -7,  2, "left",   "bottom")],
}


# ============================================================
# 5. Drawing function
# ============================================================

def draw_slope_panel(
    ax,
    data,
    ylabel,
    left_label,
    right_label,
    ylim,
    yticks,
    label_cfg,
    decimals=2,
):
    x = np.array([0, 1])

    for i in range(3):
        ax.plot(
            x,
            data[i],
            color=colors[i],
            linewidth=1.8 if i == 2 else 1.35,
            marker="o",
            markersize=5.0 if i == 2 else 4.1,
            markerfacecolor=colors[i],
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=6 - i,
        )

        for j in range(2):
            dx, dy, ha, va = label_cfg[i][j]

            ax.annotate(
                f"{data[i, j]:.{decimals}f}",
                xy=(x[j], data[i, j]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=6.6,
                color=colors[i],
                fontweight="bold",
                zorder=10,
                clip_on=False,
            )

    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label])

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    ax.set_ylabel(
        ylabel,
        fontsize=7.0,
        labelpad=3,
    )

    ax.grid(
        axis="y",
        linestyle=(0, (3, 3)),
        linewidth=0.6,
        color="#D9D9D9",
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        direction="out",
        pad=2,
    )


# ============================================================
# 6. Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
)


# Accuracy
draw_slope_panel(
    axes[0],
    accuracy,
    "Accuracy (%)",
    "Top-1",
    "Top-10",
    (10, 32),
    [10, 20, 30],
    accuracy_label_cfg,
)

# MCES
draw_slope_panel(
    axes[1],
    mces,
    "MCES",
    "MCES@1",
    "MCES@10",
    (8, 30),
    [10, 20, 30],
    mces_label_cfg,
)

# ECFP4
draw_slope_panel(
    axes[2],
    ecfp4,
    "Tanimoto",
    "ECFP4@1",
    "ECFP4@10",
    (0.25, 0.58),
    [0.3, 0.4, 0.5],
    ecfp4_label_cfg,
    decimals=2,
)


# ============================================================
# 7. Legend
# ============================================================
#
# 图例放在整个图最上方
#
# 主要调这里:
#   bbox_to_anchor=(0.5, 1.17)
#
# 第二个值越大，图例越往上
# ============================================================

legend_handles = [
    Line2D(
        [0], [0],
        color=colors[0],
        marker="o",
        linewidth=1.4,
        markersize=4.0,
        label=methods[0],
    ),
    Line2D(
        [0], [0],
        color=colors[1],
        marker="o",
        linewidth=1.4,
        markersize=4.0,
        label=methods[1],
    ),
    Line2D(
        [0], [0],
        color=colors[2],
        marker="o",
        linewidth=1.8,
        markersize=5.0,
        label=methods[2],
    ),
]

legend = fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.85),
    ncol=3,
    frameon=False,
    fontsize=8,
    columnspacing=0.9,
    handlelength=2.4,
    handletextpad=0.4,
)

for text in legend.get_texts():
    text.set_fontsize(5.9)
    text.set_fontweight("normal")


# ============================================================
# 8. Layout
# ============================================================
#
# 想让三个子图更紧凑，主要改:
#   wspace
#
# wspace 越小，子图间距越小
#
# 想整体再紧一点，还可以微调:
#   left, right, top, bottom
# ============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.985,
    top=0.74,
    bottom=0.19,
    wspace=0.3,
)


# ============================================================
# 9. Save
# ============================================================

out_dir = Path("./figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "fig2b_ablation_slope_compact.pdf"
svg_path = out_dir / "fig2b_ablation_slope_compact.svg"
png_path = out_dir / "fig2b_ablation_slope_compact.png"

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    svg_path,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    png_path,
    bbox_inches="tight",
    facecolor="white",
    dpi=600,
)

plt.show()

print("Saved:")
print(pdf_path)
print(svg_path)
print(png_path)