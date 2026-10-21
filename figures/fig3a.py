# ============================================================
# Fig. 4a
# Rank-1 de novo structure prediction quality
#
# Upper panel:
#   Top-1 exact recovery -> bars
#
# Lower panel:
#   MCES@1 -> points
#
# Sorting:
#   worse -> better from left to right
#
#   Primary:
#       Top-1 accuracy (higher = better)
#
#   Secondary:
#       MCES@1 (lower = better)
#
# Design:
#   - stacked panels with shared x-axis
#   - published baselines in blue
#   - Our Model highlighted in coral red
#   - centered legend
#   - only left and bottom spines
#   - all numerical positions manually adjustable
#
# Output:
#   fig4a_rank1_accuracy_mces.pdf
#   fig4a_rank1_accuracy_mces.svg
#   fig4a_rank1_accuracy_mces.png
# ============================================================


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


# ============================================================
# 1. MANUAL CONTROL
# ============================================================
#
# Label position format:
#
#   (dx, dy, ha, va)
#
# dx / dy:
#   offsets in points
#
# Examples:
#
# above:
#   (0, 4, "center", "bottom")
#
# below:
#   (0, -5, "center", "top")
#
# right:
#   (5, 0, "left", "center")
#
# left:
#   (-5, 0, "right", "center")
# ============================================================


# ------------------------------------------------------------
# A. Figure size
# ------------------------------------------------------------

FIG_WIDTH = 4.80
FIG_HEIGHT = 3.05


# ------------------------------------------------------------
# B. Legend position
# ------------------------------------------------------------

LEGEND_POS = (
    0.50,
    1.015,
)


# ============================================================
# C. TOP-1 ACCURACY LABEL POSITIONS
# ============================================================

top1_label_cfg = {

    # --------------------------------------------------------
    # Zero exact-recovery methods
    # --------------------------------------------------------

    "Random chemical generation":
        (
            0, 4,
            "center", "bottom"
        ),

    "SMILES Transformer":
        (
            0, 4,
            "center", "bottom"
        ),

    "SELFIES Transformer":
        (
            0, 4,
            "center", "bottom"
        ),

    "MIST + MSNovelist":
        (
            0, 4,
            "center", "bottom"
        ),

    "MIST + Neuraldecipher":
        (
            0, 4,
            "center", "bottom"
        ),

    "Spec2Mol":
        (
            0, 4,
            "center", "bottom"
        ),


    # --------------------------------------------------------
    # Non-zero methods
    # --------------------------------------------------------

    # MADGEN = 1.31
    "MADGEN":
        (
            0, 1,
            "center", "bottom"
        ),

    # DiffMS = 2.30
    "DiffMS":
        (
            0, 1,
            "center", "bottom"
        ),

    # MetGenX = 2.50
    "MetGenX":
        (
            0, 1,
            "center", "bottom"
        ),
    "FRIGID":
        (
            0, 1,
            "center", "bottom"
        ),
    # MSAnchor = 2.68
    "MSAnchor":
        (
            0, 1,
            "center", "bottom"
        ),

    # MBGen = 7.58
    "MBGen":
        (
            0, 1,
            "center", "bottom"
        ),

    # Our Model = 20.10
    "Our Model":
        (
            0, 1,
            "center", "bottom"
        ),
}


# ============================================================
# D. MCES@1 LABEL POSITIONS
# ============================================================

mces1_label_cfg = {

    # SMILES Transformer = 79.39
    "SMILES Transformer":
        (
            0, -10,
            "center", "bottom"
        ),

    # MIST + MSNovelist = 45.55
    "MIST + MSNovelist":
        (
            0, 2,
            "center", "bottom"
        ),

    # SELFIES Transformer = 38.88
    "SELFIES Transformer":
        (
            0, 2,
            "center", "bottom"
        ),

    # Spec2Mol = 37.76
    "Spec2Mol":
        (
            0, 2,
            "center", "bottom"
        ),

    # MIST + Neuraldecipher = 33.19
    "MIST + Neuraldecipher":
        (
            0, 2,
            "center", "bottom"
        ),

    # MADGEN = 27.47
    "MADGEN":
        (
            0, 2,
            "center", "bottom"
        ),

    # Random chemical generation = 21.11
    "Random chemical generation":
        (
            0, 2,
            "center", "bottom"
        ),

    # MetGenX = 20.92
    "MetGenX":
        (
            0, 2,
            "center", "bottom"
        ),
    "FRIGID":
        (
            0, 2,
            "center", "bottom"
        ),
    # DiffMS = 18.45
    "DiffMS":
        (
            0, 2,
            "center", "bottom"
        ),

    # MSAnchor = 16.57
    "MSAnchor":
        (
            0, 2,
            "center", "bottom"
        ),

    # MBGen = 13.25
    "MBGen":
        (
            0, 2,
            "center", "bottom"
        ),

    # Our Model = 10.81
    "Our Model":
        (
            0, 2,
            "center", "bottom"
        ),
}


# ------------------------------------------------------------
# E. Layout
# ------------------------------------------------------------

SUBPLOT_LEFT = 0.12
SUBPLOT_RIGHT = 0.99
SUBPLOT_TOP = 0.94
SUBPLOT_BOTTOM = 0.31

PANEL_HSPACE = 0.035


# ============================================================
# 2. Global style
# ============================================================

mpl.rcParams.update({

    "font.family": "Arial",
    "font.size": 8.0,

    "axes.labelsize": 8.0,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 6.4,
    "ytick.labelsize": 7.0,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "legend.fontsize": 9,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 150,
    "savefig.dpi": 600,
})


# ============================================================
# 3. Raw data
# ============================================================

methods_raw = [

    "Random chemical generation",
    "SMILES Transformer",
    "SELFIES Transformer",
    "MIST + MSNovelist",
    "MIST + Neuraldecipher",
    "Spec2Mol",

    "DiffMS",
    "MADGEN",
    "MSAnchor",
    "MBGen",
    "MetGenX",
    "FRIGID",

    "Our Model",
]


# ------------------------------------------------------------
# Top-1 exact structure recovery (%)
# ------------------------------------------------------------

top1_raw = np.array([

    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,

    2.30,
    1.31,
    2.68,
    7.58,
    2.50,
    18.29,

    22.75,

])


# ------------------------------------------------------------
# MCES@1
# Lower is better
# ------------------------------------------------------------

mces1_raw = np.array([

    21.11,
    79.39,
    38.88,
    45.55,
    33.19,
    37.76,

    18.45,
    27.47,
    16.57,
    13.25,
    20.92,
    13.49,

    14.74,

])

# ============================================================
# 4. Short x-axis names
# ============================================================

short_name_map = {

    "Random chemical generation":
        "Random\nchem.",

    "SMILES Transformer":
        "SMILES\nTrans.",

    "SELFIES Transformer":
        "SELFIES\nTrans.",

    "MIST + MSNovelist":
        "MIST+\nMSNovelist",

    "MIST + Neuraldecipher":
        "MIST+\nNeuraldecipher",

    "Spec2Mol":
        "Spec2Mol",

    "MADGEN":
        "MADGEN",

    "DiffMS":
        "DiffMS",

    "MetGenX":
        "MetGenX",

    "MSAnchor":
        "MSAnchor",

    "MBGen":
        "MBGen",

    "FRIGID":
        "FRIGID",

    "Our Model":
        "Our Model",
}


# ============================================================
# 5. Sort
# ============================================================
#
# We want:
#
#   LEFT  = worse
#   RIGHT = better
#
# Top-1:
#   higher = better
#
# MCES@1:
#   lower = better
#
# Primary:
#   Top-1 ascending
#
# Secondary:
#   MCES@1 descending
#
# Using -MCES means that among equal Top-1 accuracy,
# larger/worse MCES is placed further left.
# ============================================================

sort_idx = sorted(

    range(len(methods_raw)),

    key=lambda i: (
        top1_raw[i],
        -mces1_raw[i],
    )
)


methods = [
    methods_raw[i]
    for i in sort_idx
]


top1 = np.array([
    top1_raw[i]
    for i in sort_idx
])


mces1 = np.array([
    mces1_raw[i]
    for i in sort_idx
])


short_methods = [
    short_name_map[m]
    for m in methods
]


# ============================================================
# 6. Colors
# ============================================================

BASE_BAR = "#A8C1E8"
BASE_POINT = "#5D86C7"

OURS_BAR = "#F1A08B"
OURS_POINT = "#E4573D"

GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#303030"


bar_colors = [

    (
        OURS_BAR
        if method == "Our Model"
        else BASE_BAR
    )

    for method in methods
]


point_colors = [

    (
        OURS_POINT
        if method == "Our Model"
        else BASE_POINT
    )

    for method in methods
]


# ============================================================
# 7. Figure
# ============================================================

fig = plt.figure(

    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT,
    )
)


gs = fig.add_gridspec(

    nrows=2,
    ncols=1,

    height_ratios=[
        1.15,
        0.85,
    ],

    hspace=PANEL_HSPACE,
)


ax_top = fig.add_subplot(
    gs[0]
)


ax_bottom = fig.add_subplot(
    gs[1],
    sharex=ax_top,
)


# ============================================================
# 8. X positions
# ============================================================

x = np.arange(
    len(methods)
)


# ============================================================
# 9. TOP PANEL
# Top-1 accuracy bars
# ============================================================

BAR_WIDTH = 0.58


ax_top.bar(

    x,
    top1,

    width=BAR_WIDTH,

    color=bar_colors,

    edgecolor="none",

    zorder=3,
)


# ============================================================
# 10. Top-1 labels
# ============================================================

for i, (
    method,
    value,
) in enumerate(
    zip(
        methods,
        top1,
    )
):

    dx, dy, ha, va = (
        top1_label_cfg[
            method
        ]
    )


    is_ours = (
        method == "Our Model"
    )


    if value == 0:

        value_text = "0"

    else:

        value_text = (
            f"{value:.2f}"
        )


    ax_top.annotate(

        value_text,

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

        fontsize=6.6,

        color=(
            OURS_POINT
            if is_ours
            else TEXT_COLOR
        ),

        fontweight=(
            "bold"
            if is_ours
            else "normal"
        ),

        clip_on=False,

        zorder=6,
    )


# ============================================================
# 11. Top axes
# ============================================================

ax_top.set_ylim(
    0,
    25,
)


ax_top.set_yticks([
    0,
    5,
    10,
    15,
    20,
    25,
])


ax_top.set_ylabel(

    "Top-1 accuracy (%)",

    fontsize=7.2,

    fontweight="bold",

    labelpad=5,
)


ax_top.tick_params(

    axis="x",

    bottom=False,
    labelbottom=False,
)


# ============================================================
# 12. Top grid
# ============================================================

ax_top.grid(

    axis="y",

    linestyle=(
        0,
        (3, 3),
    ),

    linewidth=0.60,

    color=GRID_COLOR,
)


ax_top.set_axisbelow(
    True
)


# ============================================================
# 13. Top frame
# Only left and bottom
# ============================================================

ax_top.spines["left"].set_visible(True)
ax_top.spines["bottom"].set_visible(True)

ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)

ax_top.spines["left"].set_linewidth(0.8)
ax_top.spines["bottom"].set_linewidth(0.8)


# ============================================================
# 14. BOTTOM PANEL
# MCES@1 points
# ============================================================

for i, (
    method,
    value,
) in enumerate(
    zip(
        methods,
        mces1,
    )
):

    is_ours = (
        method == "Our Model"
    )


    ax_bottom.scatter(

        x[i],
        value,

        s=(
            48
            if is_ours
            else 34
        ),

        marker="o",

        facecolor=point_colors[i],

        edgecolor="white",

        linewidth=0.60,

        zorder=5,
    )


# ============================================================
# 15. MCES@1 labels
# ============================================================

for i, (
    method,
    value,
) in enumerate(
    zip(
        methods,
        mces1,
    )
):

    dx, dy, ha, va = (
        mces1_label_cfg[
            method
        ]
    )


    is_ours = (
        method == "Our Model"
    )


    # MCES@10 lower is better
    best_mces = np.min(mces1)


    is_best_mces = (
        value == best_mces
    )


    ax_bottom.annotate(

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

        fontsize=6.8,

        color=(
            OURS_POINT
            if is_ours
            else TEXT_COLOR
        ),

        fontweight=(
            "bold"
            if is_best_mces
            else "normal"
        ),

        clip_on=False,

        zorder=6,
    )


# ============================================================
# 16. Bottom axes
# ============================================================

ax_bottom.set_ylim(
    0,
    86,
)


ax_bottom.set_yticks([
    0,
    20,
    40,
    60,
    80,
])


ax_bottom.set_ylabel(

    "MCES@1",

    fontsize=7.2,

    fontweight="bold",

    labelpad=5,
)


# ============================================================
# 17. Shared x labels
# ============================================================

ax_bottom.set_xticks(
    x
)


ax_bottom.set_xticklabels(

    short_methods,

    rotation=32,

    ha="right",

    rotation_mode="anchor",
)


# ============================================================
# 18. Highlight Our Model
# ============================================================

for tick, method in zip(
    ax_bottom.get_xticklabels(),
    methods,
):

    if method == "Our Model":

        tick.set_color(
            OURS_POINT
        )

        tick.set_fontweight(
            "bold"
        )


# ============================================================
# 19. Bottom grid
# ============================================================

ax_bottom.grid(

    axis="y",

    linestyle=(
        0,
        (3, 3),
    ),

    linewidth=0.60,

    color=GRID_COLOR,
)


ax_bottom.set_axisbelow(
    True
)


# ============================================================
# 20. Bottom frame
# Only left and bottom
# ============================================================

ax_bottom.spines["left"].set_visible(True)
ax_bottom.spines["bottom"].set_visible(True)

ax_bottom.spines["top"].set_visible(False)
ax_bottom.spines["right"].set_visible(False)

ax_bottom.spines["left"].set_linewidth(0.8)
ax_bottom.spines["bottom"].set_linewidth(0.8)


# ============================================================
# 21. Shared x limits
# ============================================================

ax_top.set_xlim(
    -0.60,
    len(methods) - 0.40,
)


# ============================================================
# 22. CENTERED LEGEND
# ============================================================

legend_handles = [

    Patch(

        facecolor=BASE_BAR,

        edgecolor="none",

        label="Published baselines",
    ),


    Patch(

        facecolor=OURS_BAR,

        edgecolor="none",

        label="Our Model",
    ),
]


legend = ax_top.legend(

    handles=legend_handles,

    loc="upper center",

    bbox_to_anchor=(
        LEGEND_POS
    ),

    ncol=2,

    frameon=False,

    handlelength=1.8,

    handletextpad=0.40,

    columnspacing=1.2,

    borderaxespad=0.0,

    fontsize=11,
)


for text in (
    legend.get_texts()
):

    text.set_fontsize(
        7.5
    )


# ============================================================
# 23. Layout
# ============================================================

plt.subplots_adjust(

    left=SUBPLOT_LEFT,

    right=SUBPLOT_RIGHT,

    top=SUBPLOT_TOP,

    bottom=SUBPLOT_BOTTOM,
)


# ============================================================
# 24. Save
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
    "fig4a_rank1_accuracy_mces.pdf"
)


svg_path = (
    out_dir /
    "fig4a_rank1_accuracy_mces.svg"
)


png_path = (
    out_dir /
    "fig4a_rank1_accuracy_mces.png"
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


# ============================================================
# 25. Print manual coordinates
# ============================================================

print("\n===================================")
print("Top-1 accuracy label positions")
print("===================================")


for method in methods:

    idx = methods.index(method)

    print(
        f"{method:30s} "
        f"value={top1[idx]:6.2f} "
        f"position={top1_label_cfg[method]}"
    )


print("\n===================================")
print("MCES@1 label positions")
print("===================================")


for method in methods:

    idx = methods.index(method)

    print(
        f"{method:30s} "
        f"value={mces1[idx]:6.2f} "
        f"position={mces1_label_cfg[method]}"
    )


print("\nLegend position:")
print(
    "LEGEND_POS =",
    LEGEND_POS
)


print("\nSorted methods (worst -> best):")


for method, acc, mces in zip(
    methods,
    top1,
    mces1,
):

    print(
        f"{method:30s} "
        f"Top-1={acc:6.2f}   "
        f"MCES@1={mces:6.2f}"
    )


print("\nSaved files:")
print(pdf_path)
print(svg_path)
print(png_path)