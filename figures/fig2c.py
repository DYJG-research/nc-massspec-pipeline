# ============================================================
# Fig. 3g
# Ablation comparison on formula accuracy
#
# Comparison:
#   - Our Model
#   - Our Model (w/o XGB + w/o LLM)
#
# Output:
#   fig3g_ablation_formula_accuracy.pdf
#   fig3g_ablation_formula_accuracy.svg
#   fig3g_ablation_formula_accuracy.png
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# 1. Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,

    "axes.labelsize": 8.0,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,

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
# 2. Data
# ============================================================

k_labels = [
    "Top-1",
    "Top-3",
    "Top-5",
]

x = np.arange(len(k_labels))


ours = np.array([
    63.87,
    77.34,
    81.44,
])

ablation = np.array([
    33.32,
    43.08,
    48.14,
])


# ============================================================
# 3. Colors
# ============================================================

OURS_COLOR = "#C92A2A"
ABLATION_COLOR = "#8A8A8A"


# ============================================================
# 4. Figure
# ============================================================

FIG3G_WIDTH = 3.55
FIG3G_HEIGHT = 2.45

fig, ax = plt.subplots(
    figsize=(
        FIG3G_WIDTH,
        FIG3G_HEIGHT,
    )
)


# ============================================================
# 5. Ablation line
# ============================================================

ax.plot(
    x,
    ablation,

    linestyle=(0, (4, 2.5)),

    linewidth=1.45,

    marker="o",

    markersize=4.8,

    markerfacecolor="white",

    markeredgecolor=ABLATION_COLOR,

    markeredgewidth=1.1,

    color=ABLATION_COLOR,

    label="Our Model (w/o XGB + LLM)",

    zorder=3,
)


# ============================================================
# 6. Our model line
# ============================================================

ax.plot(
    x,
    ours,

    linestyle="-",

    linewidth=2.0,

    marker="o",

    markersize=5.2,

    markerfacecolor=OURS_COLOR,

    markeredgecolor="white",

    markeredgewidth=0.65,

    color=OURS_COLOR,

    label="Our Model",

    zorder=5,
)


# ============================================================
# 7. Manual label positions
# ============================================================

# Our model labels
ours_label_cfg = {
    0: (0, 5, "center", "bottom"),   # 63.87
    1: (0, 5, "center", "bottom"),   # 77.34
    2: (0, 5, "center", "bottom"),   # 81.44
}

# Ablation labels
ablation_label_cfg = {
    0: (0, 5, "center", "bottom"),   # 33.32
    1: (0, 5, "center", "bottom"),   # 43.08
    2: (0, 5, "center", "bottom"),   # 48.14
}


# ============================================================
# 8. Draw value labels
# ============================================================

for i, value in enumerate(ours):

    dx, dy, ha, va = ours_label_cfg[i]

    ax.annotate(
        f"{value:.2f}",

        xy=(
            x[i],
            value,
        ),

        xytext=(
            dx,
            dy,
        ),

        textcoords="offset points",

        ha=ha,
        va=va,

        fontsize=7,

        color=OURS_COLOR,

        clip_on=False,

        zorder=8,
    )


for i, value in enumerate(ablation):

    dx, dy, ha, va = ablation_label_cfg[i]

    ax.annotate(
        f"{value:.2f}",

        xy=(
            x[i],
            value,
        ),

        xytext=(
            dx,
            dy,
        ),

        textcoords="offset points",

        ha=ha,
        va=va,

        fontsize=7,

        color=ABLATION_COLOR,

        clip_on=False,

        zorder=8,
    )


# ============================================================
# 9. Axes
# ============================================================

ax.set_xlim(
    -0.22,
    2.22,
)

ax.set_ylim(
    25,
    86,
)

ax.set_xticks(x)

ax.set_xticklabels(
    k_labels
)

ax.set_yticks([
    30,
    40,
    50,
    60,
    70,
    80,
])

ax.set_ylabel(
    "Formula accuracy (%)",
    fontweight="bold",
    labelpad=4,
)


# ============================================================
# 10. Grid
# ============================================================

ax.grid(
    axis="y",

    linestyle=(0, (3, 3)),

    linewidth=0.65,

    color="#D9D9D9",
)

ax.set_axisbelow(True)


# ============================================================
# 11. Clean frame
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 12. Legend
# ============================================================
handles, labels = ax.get_legend_handles_labels()

# Reverse order so that Our Model appears first
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]

legend = ax.legend(
    handles,
    labels,

    loc="lower right",

    bbox_to_anchor=(
        0.985,
        0.02,
    ),

    ncol=1,

    frameon=False,

    handlelength=2.0,

    handletextpad=0.45,

    labelspacing=0.35,

    borderaxespad=0.0,

    fontsize=6.6,
)

for text in legend.get_texts():
    text.set_fontsize(6.6)
    text.set_fontweight("normal")


# ============================================================
# 13. Layout
# ============================================================

plt.subplots_adjust(
    left=0.15,
    right=0.97,

    top=0.95,
    bottom=0.20,
)


# ============================================================
# 14. Save
# ============================================================

out_dir = Path(
    "./figures"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

pdf_path = (
    out_dir /
    "fig2dd_ablation_formula_accuracy.pdf"
)

svg_path = (
    out_dir /
    "fig2dd__ablation_formula_accuracy.svg"
)

png_path = (
    out_dir /
    "fig2dd__ablation_formula_accuracy.png"
)

fig.savefig(
    pdf_path,

    bbox_inches="tight",

    pad_inches=0.05,

    facecolor="white",
)

fig.savefig(
    svg_path,

    bbox_inches="tight",

    pad_inches=0.05,

    facecolor="white",
)

fig.savefig(
    png_path,

    bbox_inches="tight",

    pad_inches=0.05,

    facecolor="white",

    dpi=600,
)

plt.show()

print("Saved files:")
print(pdf_path)
print(svg_path)
print(png_path)