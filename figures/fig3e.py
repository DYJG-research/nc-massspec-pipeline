# ============================================================
# Fig. 4c
# ECFP4 Tanimoto similarity of de novo predictions
#
# Refined horizontal paired dumbbell plot
#
# Visual encoding:
#   - Filled circle   = ECFP4 Tanimoto@1
#   - Hollow diamond = ECFP4 Tanimoto@10
#
# Method encoding:
#   - Each baseline has its own color and marker
#   - Larger method marker strip on the left
#   - Metric legend moved to upper-right
#   - Method legend placed directly below metric legend
#
# Updated:
#   - Both legends use the same larger font size
#   - Larger method identity markers
#   - Right-side empty space used more efficiently
#   - All values use two decimals
#   - SMILES Transformer 0.03 stays above its point
#
# Output:
#   fig4c_denovo_ecfp4_dumbbell_refined.pdf
#   fig4c_denovo_ecfp4_dumbbell_refined.svg
#   fig4c_denovo_ecfp4_dumbbell_refined.png
# ============================================================


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba


# ============================================================
# 1. MANUAL CONTROL
# ============================================================
#
# Numerical-label format:
#
#   (dx, dy, ha, va)
#
# dx / dy:
#   offset in points
#
# Examples:
#
# left:
#   (-5, 0, "right", "center")
#
# right:
#   (5, 0, "left", "center")
#
# above:
#   (0, 5, "center", "bottom")
#
# below:
#   (0, -5, "center", "top")
# ============================================================


# ------------------------------------------------------------
# A. Figure size
# ------------------------------------------------------------

FIG_WIDTH = 4.45
FIG_HEIGHT = 3.05


# ------------------------------------------------------------
# B. Legend font size
#
# Both legends use exactly the same font size.
# ------------------------------------------------------------

LEGEND_FONT_SIZE = 7.6


# ------------------------------------------------------------
# C. Metric legend position
#
# Now placed in upper-right.
#
# (x, y)
# ------------------------------------------------------------

METRIC_LEGEND_POS = (
    0.995,
    0.995,
)


# ------------------------------------------------------------
# D. Method legend position
#
# Directly below metric legend.
# ------------------------------------------------------------

METHOD_LEGEND_POS = (
    0.995,
    0.865,
)


# ------------------------------------------------------------
# E. Method marker strip position
# ------------------------------------------------------------

METHOD_MARKER_X = 0.50


# ------------------------------------------------------------
# F. Marker sizes
# ------------------------------------------------------------
#
# Increased relative to previous version.
# ------------------------------------------------------------

METHOD_MARKER_SIZE = 52
OURS_METHOD_MARKER_SIZE = 62

METHOD_LEGEND_MARKER_SIZE = 6.0
METRIC_LEGEND_MARKER_SIZE = 6.1


# ------------------------------------------------------------
# G. X-axis range
# ------------------------------------------------------------

X_MIN = 0.00
X_MAX = 0.63


# ============================================================
# H. Tanimoto@1 LABEL POSITIONS
# ============================================================

ecfp1_label_cfg = {

    # SMILES Transformer = 0.03
    # Above point to avoid y-axis overlap
    "SMILES Transformer":
        (
            0, 5,
            "center", "bottom"
        ),

    "Random chemical generation":
        (
            -5, 0,
            "right", "center"
        ),

    "SELFIES Transformer":
        (
            -5, 0,
            "right", "center"
        ),

    "MIST + MSNovelist":
        (
            -5, 0,
            "right", "center"
        ),

    "MIST + Neuraldecipher":
        (
            -5, 0,
            "right", "center"
        ),

    "Spec2Mol":
        (
            -5, 0,
            "right", "center"
        ),

    "MADGEN":
        (
            -5, 0,
            "right", "center"
        ),

    "MetGenX":
        (
            -5, 0,
            "right", "center"
        ),
    "FRIGID":
        (
            -5, 0,
            "right", "center"
        ),
    "DiffMS":
        (
            -5, 0,
            "right", "center"
        ),

    "MSAnchor":
        (
            -5, 0,
            "right", "center"
        ),

    "MBGen":
        (
            -5, 0,
            "right", "center"
        ),

    "Our Model":
        (
            -6, 0,
            "right", "center"
        ),
}


# ============================================================
# I. Tanimoto@10 LABEL POSITIONS
# ============================================================

ecfp10_label_cfg = {

    "SMILES Transformer":
        (
            5, 0,
            "left", "center"
        ),

    "Random chemical generation":
        (
            5, 0,
            "left", "center"
        ),

    "SELFIES Transformer":
        (
            5, 0,
            "left", "center"
        ),

    "MIST + MSNovelist":
        (
            5, 0,
            "left", "center"
        ),

    "MIST + Neuraldecipher":
        (
            5, 0,
            "left", "center"
        ),

    "Spec2Mol":
        (
            5, 0,
            "left", "center"
        ),

    "MADGEN":
        (
            5, 0,
            "left", "center"
        ),

    "MetGenX":
        (
            5, 0,
            "left", "center"
        ),
    "FRIGID":
        (
            5, 0,
            "left", "center"
        ),
    "DiffMS":
        (
            5, 0,
            "left", "center"
        ),

    "MSAnchor":
        (
            5, 0,
            "left", "center"
        ),

    "MBGen":
        (
            5, 0,
            "left", "center"
        ),

    "Our Model":
        (
            6, 0,
            "left", "center"
        ),
}


# ------------------------------------------------------------
# J. Layout
# ------------------------------------------------------------

SUBPLOT_LEFT = 0.045
SUBPLOT_RIGHT = 0.985
SUBPLOT_TOP = 0.94
SUBPLOT_BOTTOM = 0.16


# ============================================================
# 2. GLOBAL STYLE
# ============================================================

mpl.rcParams.update({

    "font.family": "Arial",
    "font.size": 8.0,

    "axes.labelsize": 7.4,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 7.0,
    "ytick.labelsize": 6.8,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.major.size": 3,
    "ytick.major.size": 0,

    "legend.fontsize": LEGEND_FONT_SIZE,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 150,
    "savefig.dpi": 600,
})


# ============================================================
# 3. RAW DATA
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


ecfp1_raw = np.array([

    0.08,
    0.03,
    0.08,
    0.06,
    0.14,
    0.12,

    0.28,
    0.20,
    0.32,
    0.38,
    0.22,
    0.43,

    0.45,
])


ecfp10_raw = np.array([

    0.11,
    0.10,
    0.13,
    0.15,
    0.16,
    0.16,

    0.39,
    0.26,
    0.41,
    0.47,
    0.28,
    0.47,

    0.53,
])


# ============================================================
# 4. SORT
# ============================================================
#
# Higher similarity = better.
#
# Worse methods at top,
# better methods at bottom.
# ============================================================

sort_idx = sorted(

    range(len(methods_raw)),

    key=lambda i: (
        ecfp10_raw[i],
        ecfp1_raw[i],
    )
)


methods = [
    methods_raw[i]
    for i in sort_idx
]


ecfp1 = np.array([
    ecfp1_raw[i]
    for i in sort_idx
])


ecfp10 = np.array([
    ecfp10_raw[i]
    for i in sort_idx
])


# ============================================================
# 5. METHOD COLORS
# ============================================================

method_color_map = {

    "Random chemical generation":
        "#8A8A8A",

    "SMILES Transformer":
        "#7C6BB3",

    "SELFIES Transformer":
        "#6FA8DC",

    "MIST + MSNovelist":
        "#3FA59B",

    "MIST + Neuraldecipher":
        "#73B66B",

    "Spec2Mol":
        "#C97AA6",

    "MADGEN":
        "#D9903D",

    "MetGenX":
        "#B78BA4",

    "DiffMS":
        "#557FB8",

    "MSAnchor":
        "#4F9B8A",

    "MBGen":
        "#A47C62",

    "FRIGID":
        "#110EC7",

    "Our Model":
        "#C92A2A",
}


# ============================================================
# 6. METHOD MARKERS
# ============================================================

method_marker_map = {

    "Random chemical generation":
        "o",

    "SMILES Transformer":
        "^",

    "SELFIES Transformer":
        "s",

    "MIST + MSNovelist":
        "D",

    "MIST + Neuraldecipher":
        "v",

    "Spec2Mol":
        "P",

    "MADGEN":
        "X",

    "MetGenX":
        "<",

    "DiffMS":
        ">",

    "MSAnchor":
        "h",

    "MBGen":
        "p",
    "FRIGID":
        "o",
    "Our Model":
        "*",
}


# ============================================================
# 7. METHOD LEGEND NAMES
# ============================================================

method_legend_name = {

    "Random chemical generation":
        "Random generation",

    "SMILES Transformer":
        "SMILES Transformer",

    "SELFIES Transformer":
        "SELFIES Transformer",

    "MIST + MSNovelist":
        "MIST + MSNovelist",

    "MIST + Neuraldecipher":
        "MIST + Neuraldecipher",

    "Spec2Mol":
        "Spec2Mol",

    "MADGEN":
        "MADGEN",

    "MetGenX":
        "MetGenX",

    "DiffMS":
        "DiffMS",

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
# 8. FIGURE
# ============================================================

fig = plt.figure(

    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT,
    )
)


# ------------------------------------------------------------
# Slightly wider marker strip to support larger symbols
# ------------------------------------------------------------

gs = GridSpec(

    nrows=1,
    ncols=2,

    width_ratios=[
        0.060,
        0.940,
    ],

    wspace=0.006,

    figure=fig,
)


ax_method = fig.add_subplot(
    gs[0, 0]
)


ax = fig.add_subplot(
    gs[0, 1],
    sharey=ax_method,
)


# ============================================================
# 9. Y POSITIONS
# ============================================================

y = np.arange(
    len(methods)
)[::-1]


# ============================================================
# 10. DRAW DUMBBELLS
# ============================================================

for (
    yi,
    method,
    value1,
    value10,
) in zip(
    y,
    methods,
    ecfp1,
    ecfp10,
):

    color = method_color_map[
        method
    ]


    is_ours = (
        method == "Our Model"
    )


    # --------------------------------------------------------
    # Connecting line
    # --------------------------------------------------------

    ax.hlines(

        y=yi,

        xmin=value1,
        xmax=value10,

        color=to_rgba(
            color,
            0.68 if not is_ours else 0.88
        ),

        linewidth=(
            2.0
            if is_ours
            else 1.55
        ),

        zorder=2,
    )


    # --------------------------------------------------------
    # Tanimoto@1
    # Filled circle
    # --------------------------------------------------------

    ax.scatter(

        value1,
        yi,

        s=(
            48
            if is_ours
            else 38
        ),

        marker="o",

        facecolor=color,

        edgecolor="white",

        linewidth=0.65,

        zorder=5,
    )


    # --------------------------------------------------------
    # Tanimoto@10
    # Hollow diamond
    # --------------------------------------------------------

    ax.scatter(

        value10,
        yi,

        s=(
            54
            if is_ours
            else 43
        ),

        marker="D",

        facecolor="white",

        edgecolor=color,

        linewidth=(
            1.40
            if is_ours
            else 1.15
        ),

        zorder=5,
    )


# ============================================================
# Best values
# Higher ECFP4 Tanimoto is better
# ============================================================

best_ecfp1 = np.max(ecfp1)

best_ecfp10 = np.max(ecfp10)

# ============================================================
# 11. NUMERICAL LABELS
# ============================================================

for (
    yi,
    method,
    value1,
    value10,
) in zip(
    y,
    methods,
    ecfp1,
    ecfp10,
):

    color = method_color_map[
        method
    ]


    is_ours = (
        method == "Our Model"
    )

    is_best_ecfp1 = (
        value1 == best_ecfp1
    )


    is_best_ecfp10 = (
        value10 == best_ecfp10
    )

    # --------------------------------------------------------
    # Tanimoto@1
    # --------------------------------------------------------

    dx, dy, ha, va = (
        ecfp1_label_cfg[
            method
        ]
    )


    ax.annotate(

        f"{value1:.2f}",

        xy=(
            value1,
            yi,
        ),

        xytext=(
            dx,
            dy,
        ),

        textcoords="offset points",

        ha=ha,
        va=va,

        fontsize=6.3,

        color=(
            color
            if is_ours
            else "#303030"
        ),

        fontweight=(
            "bold"
            if is_best_ecfp1
            else "normal"
        ),

        clip_on=False,

        zorder=8,
    )


    # --------------------------------------------------------
    # Tanimoto@10
    # --------------------------------------------------------

    dx, dy, ha, va = (
        ecfp10_label_cfg[
            method
        ]
    )


    ax.annotate(

        f"{value10:.2f}",

        xy=(
            value10,
            yi,
        ),

        xytext=(
            dx,
            dy,
        ),

        textcoords="offset points",

        ha=ha,
        va=va,

        fontsize=6.3,

        color=color,

        fontweight=(
            "bold"
            if is_best_ecfp10
            else "normal"
        ),

        clip_on=False,

        zorder=8,
    )


# ============================================================
# 12. LEFT METHOD MARKER STRIP
# ============================================================

ax_method.set_xlim(
    0,
    1,
)


for (
    yi,
    method,
) in zip(
    y,
    methods,
):

    color = method_color_map[
        method
    ]


    is_ours = (
        method == "Our Model"
    )


    ax_method.scatter(

        METHOD_MARKER_X,
        yi,

        s=(
            OURS_METHOD_MARKER_SIZE
            if is_ours
            else METHOD_MARKER_SIZE
        ),

        marker=method_marker_map[
            method
        ],

        facecolor=color,

        edgecolor="white",

        linewidth=0.60,

        clip_on=False,

        zorder=10,
    )


ax_method.set_xticks([])

ax_method.set_yticks(
    y
)

ax_method.set_yticklabels(
    [""] * len(methods)
)


ax_method.tick_params(

    left=False,
    right=False,

    bottom=False,
    top=False,
)


for spine in (
    ax_method.spines.values()
):

    spine.set_visible(
        False
    )


# ============================================================
# 13. MAIN AXIS
# ============================================================

ax.set_xlim(
    X_MIN,
    X_MAX,
)


ax.set_xticks([
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
])


ax.set_yticks(
    y
)


ax.set_yticklabels(
    [""] * len(methods)
)


ax.tick_params(
    axis="y",
    length=0,
)


ax.set_xlabel(

    "ECFP4 Tanimoto similarity",

    fontsize=7.8,

    fontweight="bold",

    labelpad=4,
)


# ============================================================
# 14. GRID
# ============================================================

ax.grid(

    axis="x",

    linestyle=(
        0,
        (3, 3),
    ),

    linewidth=0.58,

    color="#D9D9D9",
)


ax.set_axisbelow(
    True
)


# ============================================================
# 15. FRAME
# ============================================================

ax.spines["left"].set_visible(
    True
)

ax.spines["bottom"].set_visible(
    True
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)


ax.spines["left"].set_linewidth(
    0.8
)

ax.spines["bottom"].set_linewidth(
    0.8
)


# ============================================================
# 16. METRIC LEGEND
#
# Now in the upper-right.
# ============================================================

metric_legend_handles = [

    Line2D(

        [0], [0],

        marker="o",

        linestyle="none",

        markersize=METRIC_LEGEND_MARKER_SIZE,

        markerfacecolor="#555555",

        markeredgecolor="white",

        markeredgewidth=0.60,

        label="Tanimoto@1",
    ),


    Line2D(

        [0], [0],

        marker="D",

        linestyle="none",

        markersize=METRIC_LEGEND_MARKER_SIZE,

        markerfacecolor="white",

        markeredgecolor="#555555",

        markeredgewidth=1.20,

        label="Tanimoto@10",
    ),
]


metric_legend = ax.legend(

    handles=metric_legend_handles,

    loc="upper right",

    bbox_to_anchor=(
        METRIC_LEGEND_POS
    ),

    ncol=2,

    frameon=False,

    handlelength=1.0,

    handletextpad=0.36,

    columnspacing=0.95,

    labelspacing=0.25,

    borderaxespad=0.0,

    fontsize=LEGEND_FONT_SIZE,
)


for text in (
    metric_legend.get_texts()
):

    text.set_fontsize(
        LEGEND_FONT_SIZE
    )


ax.add_artist(
    metric_legend
)


# ============================================================
# 17. METHOD LEGEND
#
# Manually split into two legends:
#
# Left column:
#   6 methods
#
# Right column:
#   7 methods
#
# This avoids matplotlib automatic column imbalance.
# ============================================================


method_legend_handles = []


for method in methods:

    color = method_color_map[
        method
    ]


    method_legend_handles.append(

        Line2D(

            [0], [0],

            marker=method_marker_map[
                method
            ],

            linestyle="none",

            markersize=METHOD_LEGEND_MARKER_SIZE,

            markerfacecolor=color,

            markeredgecolor="white",

            markeredgewidth=0.50,

            label=method_legend_name[
                method
            ],
        )
    )



# ------------------------------------------------------------
# Split methods manually
# ------------------------------------------------------------

left_methods = [

    "Random chemical generation",

    "SMILES Transformer",

    "SELFIES Transformer",

    "MIST + MSNovelist",

    "MIST + Neuraldecipher",

    "Spec2Mol",

]


right_methods = [

    "DiffMS",

    "MADGEN",

    "MSAnchor",

    "MBGen",

    "MetGenX",

    "FRIGID",

    "Our Model",

]


left_handles = []
left_labels = []

right_handles = []
right_labels = []


for handle, method in zip(
    method_legend_handles,
    methods
):

    if method in left_methods:

        left_handles.append(
            handle
        )

        left_labels.append(
            method_legend_name[method]
        )


    elif method in right_methods:

        right_handles.append(
            handle
        )

        right_labels.append(
            method_legend_name[method]
        )



# ------------------------------------------------------------
# Left column legend
# ------------------------------------------------------------

method_legend_left = ax.legend(

    handles=left_handles,

    labels=left_labels,

    loc="upper left",

    bbox_to_anchor=(

        0.5,

        0.865,

    ),

    frameon=False,

    handlelength=0.80,

    handletextpad=0.32,

    labelspacing=0.28,

    borderaxespad=0.0,

    fontsize=LEGEND_FONT_SIZE,
)


for text in (
    method_legend_left.get_texts()
):

    text.set_fontsize(
        LEGEND_FONT_SIZE
    )


ax.add_artist(
    method_legend_left
)



# ------------------------------------------------------------
# Right column legend
# ------------------------------------------------------------

method_legend_right = ax.legend(

    handles=right_handles,

    labels=right_labels,

    loc="upper left",

    bbox_to_anchor=(

        0.82,

        0.865,

    ),

    frameon=False,

    handlelength=0.80,

    handletextpad=0.32,

    labelspacing=0.28,

    borderaxespad=0.0,

    fontsize=LEGEND_FONT_SIZE,
)


for text in (
    method_legend_right.get_texts()
):

    text.set_fontsize(
        LEGEND_FONT_SIZE
    )

# ============================================================
# 18. Y LIMITS
# ============================================================

ax.set_ylim(

    -0.35,

    len(methods) - 0.65,
)


ax_method.set_ylim(

    -0.35,

    len(methods) - 0.65,
)


# ============================================================
# 19. LAYOUT
# ============================================================

plt.subplots_adjust(

    left=SUBPLOT_LEFT,

    right=SUBPLOT_RIGHT,

    top=SUBPLOT_TOP,

    bottom=SUBPLOT_BOTTOM,
)


# ============================================================
# 20. SAVE
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
    "fig4c_denovo_ecfp4_dumbbell_refined.pdf"
)


svg_path = (
    out_dir /
    "fig4c_denovo_ecfp4_dumbbell_refined.svg"
)


png_path = (
    out_dir /
    "fig4c_denovo_ecfp4_dumbbell_refined.png"
)


fig.savefig(

    pdf_path,

    bbox_inches="tight",

    pad_inches=0.04,

    facecolor="white",
)


fig.savefig(

    svg_path,

    bbox_inches="tight",

    pad_inches=0.04,

    facecolor="white",
)


fig.savefig(

    png_path,

    bbox_inches="tight",

    pad_inches=0.04,

    facecolor="white",

    dpi=600,
)


plt.show()


# ============================================================
# 21. PRINT MANUAL POSITIONS
# ============================================================

print("\n===================================")
print("Tanimoto@1 label positions")
print("===================================")


for method in methods:

    idx = methods.index(
        method
    )

    print(
        f"{method:30s} "
        f"value={ecfp1[idx]:6.2f} "
        f"position={ecfp1_label_cfg[method]}"
    )


print("\n===================================")
print("Tanimoto@10 label positions")
print("===================================")


for method in methods:

    idx = methods.index(
        method
    )

    print(
        f"{method:30s} "
        f"value={ecfp10[idx]:6.2f} "
        f"position={ecfp10_label_cfg[method]}"
    )


print("\nMetric legend:")
print(
    "METRIC_LEGEND_POS =",
    METRIC_LEGEND_POS
)


print("\nMethod legend:")
print(
    "METHOD_LEGEND_POS =",
    METHOD_LEGEND_POS
)


print("\nSorted methods (worst -> best):")


for (
    method,
    value1,
    value10,
) in zip(
    methods,
    ecfp1,
    ecfp10,
):

    print(
        f"{method:30s} "
        f"@1={value1:6.2f}   "
        f"@10={value10:6.2f}"
    )


print("\nSaved files:")
print(pdf_path)
print(svg_path)
print(png_path)