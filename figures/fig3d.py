# ============================================================
# Formula-hypothesis trend analysis
#
# 2 x 3 small-multiple line charts
#
# Panels:
#   1. Top-1 accuracy
#   2. Top-10 accuracy
#   3. MCES@1
#   4. MCES@10
#   5. ECFP4@1
#   6. ECFP4@10
#
# Output:
#   fig_formula_hypothesis_trends.pdf
#   fig_formula_hypothesis_trends.svg
#   fig_formula_hypothesis_trends.png
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

    "axes.labelsize": 7.4,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "legend.fontsize": 6.6,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 150,
    "savefig.dpi": 600,
})


# ============================================================
# 2. Figure size
# ============================================================

FIG_WIDTH = 3.4
FIG_HEIGHT = 3.6


# ============================================================
# 3. Data
# ============================================================

x_labels = ["Top-1", "Top-3", "Top-5", "Top-10", "Ref."]
x = np.arange(len(x_labels))

top1_acc = np.array([16.50, 21.13, 22.37, 22.75, 26.21])
top10_acc = np.array([20.78, 27.10, 28.59, 29.22, 33.44])

mces1 = np.array([17.70, 15.37, 15.07, 14.74, 10.19])
mces10 = np.array([15.69, 11.65, 11.03, 10.61, 8.41])

ecfp1 = np.array([0.38, 0.43, 0.44, 0.45, 0.50])
ecfp10 = np.array([0.45, 0.51, 0.52, 0.53, 0.58])


# ============================================================
# 4. Colors
# ============================================================

MAIN_COLOR = "#D55E00"      # orange-red
GT_COLOR = "#3C3C3C"        # dark gray
GRID_COLOR = "#D9D9D9"


# ============================================================
# 5. Manual label offsets
# ============================================================
#
# 每个子图 5 个点，对应:
#   0 -> Top-1
#   1 -> Top-3
#   2 -> Top-5
#   3 -> Top-10
#   4 -> GT
#
# tuple 格式:
#   (dx, dy, ha, va)
#
# 如果你后面想手调数值位置，只改这些字典即可
# ============================================================

top1_acc_label_cfg = {
    0: (11, -4, "center", "bottom"),
    1: (0, 2, "center", "bottom"),
    2: (0, 2, "center", "bottom"),
    3: (0, -4, "center", "top"),
    4: (0, 3, "center", "bottom"),
}

top10_acc_label_cfg = {
    0: (13, -2, "center", "bottom"),
    1: (0, 2, "center", "bottom"),
    2: (0, -4, "center", "top"),
    3: (-2, 2, "center", "bottom"),
    4: (0, 3, "center", "bottom"),
}

mces1_label_cfg = {
    0: (2, 2, "center", "bottom"),
    1: (0, -4, "center", "top"),
    2: (0, 2, "center", "bottom"),
    3: (0, 2, "center", "bottom"),
    4: (-12, 2, "center", "top"),
}

mces10_label_cfg = {
    0: (2, 2, "center", "bottom"),
    1: (5, 2, "center", "bottom"),
    2: (0, -4, "center", "top"),
    3: (2, 2, "center", "bottom"),
    4: (0, 9, "center", "top"),
}

ecfp1_label_cfg = {
    0: (10, 2, "center", "top"),
    1: (0, 2, "center", "bottom"),
    2: (0, -4, "center", "top"),
    3: (-2, 2, "center", "bottom"),
    4: (0, 2, "center", "bottom"),
}

ecfp10_label_cfg = {
    0: (10, 3, "center", "top"),
    1: (0, 3, "center", "bottom"),
    2: (0, -4, "center", "top"),
    3: (-2, 2, "center", "bottom"),
    4: (0, 2, "center", "bottom"),
}


# ============================================================
# 6. Helper function
# ============================================================

def draw_metric_panel(
    ax,
    y,
    ylabel,
    ylim,
    yticks,
    label_cfg,
    decimals=2,
):
    # Main line for Top-1 to Top-10
    ax.plot(
        x[:4],
        y[:4],
        color=MAIN_COLOR,
        linewidth=1.8,
        marker="o",
        markersize=4.6,
        markerfacecolor=MAIN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=4,
    )

    # Connecting segment from Top-10 to GT
    ax.plot(
        x[3:],
        y[3:],
        color=MAIN_COLOR,
        linewidth=1.4,
        linestyle=(0, (3, 2)),
        zorder=3,
    )

    # GT point
    ax.scatter(
        x[4],
        y[4],
        s=28,
        marker="D",
        facecolor=GT_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    # Re-draw first 4 points to keep them crisp
    ax.scatter(
        x[:4],
        y[:4],
        s=26,
        marker="o",
        facecolor=MAIN_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    # Value labels
    for i, yi in enumerate(y):
        dx, dy, ha, va = label_cfg[i]

        ax.annotate(
            f"{yi:.{decimals}f}",
            xy=(x[i], yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.3,
            color=GT_COLOR if i == 4 else MAIN_COLOR,
            fontweight="bold",
            zorder=10,
            clip_on=False,
        )

    # Axes
    ax.set_xlim(-0.25, 4.25)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)

    ax.set_ylabel(
        ylabel,
        fontsize=7.2,
        labelpad=3,
    )

    # Grid
    ax.grid(
        axis="y",
        linestyle=(0, (3, 3)),
        linewidth=0.6,
        color=GRID_COLOR,
    )
    ax.set_axisbelow(True)

    # Clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        direction="out",
        pad=2,
    )


# ============================================================
# 7. Create figure
# ============================================================

fig, axes = plt.subplots(
    3, 2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
)

ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()


# ============================================================
# 8. Draw panels
# ============================================================

draw_metric_panel(
    ax=ax1,
    y=top1_acc,
    ylabel="Top-1 Acc (%)",
    ylim=(15, 28),
    yticks=[16, 20, 24, 28],
    label_cfg=top1_acc_label_cfg,
)

draw_metric_panel(
    ax=ax2,
    y=top10_acc,
    ylabel="Top-10 Acc (%)",
    ylim=(20, 35),
    yticks=[20, 25, 30, 35],
    label_cfg=top10_acc_label_cfg,
)

draw_metric_panel(
    ax=ax3,
    y=mces1,
    ylabel="MCES@1",
    ylim=(9, 18.5),
    yticks=[10, 12, 14, 16, 18],
    label_cfg=mces1_label_cfg,
)

draw_metric_panel(
    ax=ax4,
    y=mces10,
    ylabel="MCES@10",
    ylim=(8, 16.5),
    yticks=[8, 10, 12, 14, 16],
    label_cfg=mces10_label_cfg,
)

draw_metric_panel(
    ax=ax5,
    y=ecfp1,
    ylabel="ECFP4@1",
    ylim=(0.36, 0.52),
    yticks=[0.38, 0.42, 0.46, 0.50],
    label_cfg=ecfp1_label_cfg,
    decimals=2,
)

draw_metric_panel(
    ax=ax6,
    y=ecfp10,
    ylabel="ECFP4@10",
    ylim=(0.44, 0.60),
    yticks=[0.45, 0.50, 0.55, 0.60],
    label_cfg=ecfp10_label_cfg,
    decimals=2,
)


# ============================================================
# 9. Shared legend
# ============================================================

legend_handles = [
    Line2D(
        [0], [0],
        color=MAIN_COLOR,
        marker="o",
        linewidth=1.8,
        markersize=4.6,
        markerfacecolor=MAIN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Predicted formula set",
    ),
    Line2D(
        [0], [0],
        color=MAIN_COLOR,
        linestyle=(0, (3, 2)),
        marker="D",
        linewidth=1.4,
        markersize=4.8,
        markerfacecolor=GT_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="Reference formula",
    ),
]

legend = fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=2,
    frameon=False,
    columnspacing=1.2,
    handlelength=2.4,
    handletextpad=0.5,
    fontsize=8,
)

for text in legend.get_texts():
    text.set_fontsize(6.6)
    text.set_fontweight("normal")


# ============================================================
# 10. Layout
# ============================================================

plt.subplots_adjust(
    left=0.09,
    right=0.985,
    top=0.83,
    bottom=0.11,
    wspace=0.42,
    hspace=0.50,
)


# ============================================================
# 11. Save
# ============================================================

out_dir = Path("./figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "fig_formula_hypothesis_trends.pdf"
svg_path = out_dir / "fig_formula_hypothesis_trends.svg"
png_path = out_dir / "fig_formula_hypothesis_trends.png"

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

print("Saved files:")
print(pdf_path)
print(svg_path)
print(png_path)