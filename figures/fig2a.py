# ============================================================
# Fig. 2a
# Overall molecular formula identification performance
#
# Three subplots:
#   Top-1 accuracy (%)
#   Top-3 accuracy (%)
#   Top-5 accuracy (%)
#
# Fig. 2a contains only the final / strongest configuration
# of each method. Ablations and control analyses are shown
# separately in subsequent panels.
#
# Output:
#   fig2a_formula_accuracy_final.pdf
#   fig2a_formula_accuracy_final.svg
#   fig2a_formula_accuracy_final.png
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import transforms


# ============================================================
# 1. Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,

    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 0,

    "legend.fontsize": 6.8,

    # Keep text editable in vector outputs
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 150,
    "savefig.dpi": 600,
})


# ============================================================
# 2. Data
# ============================================================

methods = [
    "Mass-only formula enumeration",
    "SIRIUS 6.3.7",
    "BUDDY",
    "MIST-CF",
    "FIDDLE",
    "Our Model",
]


# Final / strongest configuration of each method

top1 = np.array([
    30.48,   # Mass-only formula enumeration
    38.64,   # SIRIUS 6.3.7
    46.88,   # BUDDY
    33.29,   # MIST-CF
    23.76,   # FIDDLE
    63.87,   # Our Model
])

top3 = np.array([
    36.41,   # Mass-only formula enumeration
    57.66,   # SIRIUS 6.3.7
    56.96,   # BUDDY
    52.46,   # MIST-CF
    34.94,   # FIDDLE
    77.34,   # Our Model
])

top5 = np.array([
    37.99,   # Mass-only formula enumeration
    65.74,   # SIRIUS 6.3.7
    60.35,   # BUDDY
    59.73,   # MIST-CF
    38.29,   # FIDDLE
    81.44,   # Our Model
])


datasets = [top1, top3, top5]

xlabels = [
    "Top-1 accuracy (%)",
    "Top-3 accuracy (%)",
    "Top-5 accuracy (%)",
]


# ============================================================
# 3. Colors and markers
# ============================================================

# Each method has an independent color.
# Keep these colors unchanged in later figures whenever
# the same method reappears.

color_map = {
    "Mass-only formula enumeration": "#7F7F7F",
    "SIRIUS 6.3.7": "#3B6FB6",
    "BUDDY": "#6FA8DC",
    "MIST-CF": "#3FA59B",
    "FIDDLE": "#73B66B",
    "Our Model": "#C92A2A",
}

marker_map = {
    "Mass-only formula enumeration": "o",
    "SIRIUS 6.3.7": "^",
    "BUDDY": "s",
    "MIST-CF": "D",
    "FIDDLE": "v",
    "Our Model": "*",
}


# ============================================================
# 4. Helper: smart x-axis range
# ============================================================

def smart_axis(values):
    """
    Automatically choose a clean x-axis upper limit
    according to the maximum value of each subplot.
    """

    vmax = float(np.max(values))

    if vmax <= 50:
        upper = int(np.ceil(vmax / 5.0) * 5)
        step = 10

    elif vmax <= 70:
        upper = int(np.ceil(vmax / 5.0) * 5)
        step = 10

    else:
        upper = int(np.ceil(vmax / 5.0) * 5)
        step = 15

    # Leave space for numerical labels
    if upper - vmax < 2.5:
        upper += 5

    ticks = np.arange(
        0,
        upper + 0.1,
        step
    )

    return upper, ticks


# ============================================================
# 5. Figure canvas
# ============================================================

# Keep the spacing settings from your current version.
fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 2.0),
    sharey=True
)

# First method at top
y = np.arange(len(methods))[::-1]


# ============================================================
# 6. Draw three subplots
# ============================================================

for ax, values, xlabel in zip(
    axes,
    datasets,
    xlabels
):

    xmax, xticks = smart_axis(values)

    # --------------------------------------------------------
    # Draw each method
    # --------------------------------------------------------

    for yi, method, value in zip(
        y,
        methods,
        values
    ):

        color = color_map[method]
        marker = marker_map[method]

        # ----------------------------------------------------
        # Horizontal line
        # ----------------------------------------------------

        ax.hlines(
            y=yi,
            xmin=0,
            xmax=value,
            color=color,
            linewidth=1.25,
            alpha=0.88,
            zorder=2,
        )

        # ----------------------------------------------------
        # Endpoint marker
        # ----------------------------------------------------

        ax.scatter(
            value,
            yi,

            # Highlight our model slightly
            s=78 if method == "Our Model" else 36,

            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

                # ----------------------------------------------------
        # Numerical value
        # ----------------------------------------------------

        value_text_gap = 2.0

        if method == "Our Model":

            # Our Model:
            # place the value directly above the marker
            ax.annotate(
                f"{value:.2f}",
                xy=(value, yi),
                xytext=(0, 3),
                textcoords="offset points",

                ha="center",
                va="bottom",

                fontsize=6.8,
                color="#B42318",
                fontweight="bold",

                zorder=5,
            )

        else:

            # Other methods:
            # keep the original position on the right
            ax.text(
                value + value_text_gap,
                yi,
                f"{value:.2f}",

                ha="left",
                va="center",

                fontsize=6.8,
                color="#303030",
                fontweight="normal",

                zorder=5,
            )


    # ========================================================
    # Marker icons on the left of every subplot
    # ========================================================

    trans = transforms.blended_transform_factory(
        ax.transAxes,
        ax.transData
    )

    # Distance between left marker and y-axis.
    # Smaller absolute value -> closer to y-axis.
    marker_x = -0.04

    for yi, method in zip(
        y,
        methods
    ):

        ax.scatter(
            marker_x,
            yi,

            transform=trans,

            s=72 if method == "Our Model" else 40,

            marker=marker_map[method],
            facecolor=color_map[method],

            edgecolor="white",
            linewidth=0.6,

            clip_on=False,
            zorder=6,
        )


    # ========================================================
    # X axis
    # ========================================================

    ax.set_xlim(
        0,
        xmax
    )

    ax.set_xticks(
        xticks
    )

    # Only one title / axis label for each subplot
    ax.set_xlabel(
        xlabel,
        labelpad=4,
        fontweight="bold"
    )


    # ========================================================
    # Y axis
    # ========================================================

    ax.set_yticks(y)

    # Method names are shown only in the legend
    ax.set_yticklabels(
        [""] * len(y)
    )


    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        axis="x",
        linestyle=(0, (3, 3)),
        linewidth=0.65,
        color="#D9D9D9"
    )

    ax.set_axisbelow(True)


    # ========================================================
    # Clean journal-style frame
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 7. Shared legend above
# ============================================================

handles = []

for method in methods:
    handles.append(
        Line2D(
            [0], [0],
            color=color_map[method],
            marker=marker_map[method],
            linestyle="-",
            linewidth=1.2,

            # Make legend markers visually consistent
            # with the left-side y-axis markers
            markersize=8.8 if method == "Our Model" else 6.8,

            markerfacecolor=color_map[method],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=method,
        )
    )

legend = fig.legend(
    handles=handles,
    labels=methods,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.90),
    ncol=6,
    frameon=False,
    columnspacing=1.15,
    handlelength=2.0,
    handletextpad=0.40,
    fontsize=6.2,
)

# Force all legend text to have identical style
for text in legend.get_texts():
    text.set_fontsize(6.8)
    text.set_fontweight("normal")




# ============================================================
# 8. Layout
# ============================================================
#
# Keep the same spacing logic as your uploaded version.
#
# top:
#   larger -> plots move upward, closer to legend
#
# bottom:
#   controls space for x-axis labels
#
# wspace:
#   horizontal spacing among Top-1 / Top-3 / Top-5 panels
# ============================================================

plt.subplots_adjust(
    left=0.10,
    right=0.99,

    top=0.78,

    bottom=0.18,

    wspace=0.18,
)


# ============================================================
# 9. Save
# ============================================================

out_dir = Path("./figures")

out_dir.mkdir(
    parents=True,
    exist_ok=True
)


pdf_path = (
    out_dir /
    "fig2a_formula_accuracy_final.pdf"
)

svg_path = (
    out_dir /
    "fig2a_formula_accuracy_final.svg"
)

png_path = (
    out_dir /
    "fig2a_formula_accuracy_final.png"
)


# PDF
fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)


# SVG
fig.savefig(
    svg_path,
    bbox_inches="tight",
    facecolor="white"
)


# 600-dpi PNG
fig.savefig(
    png_path,
    bbox_inches="tight",
    facecolor="white",
    dpi=600
)


plt.show()


print("Saved files:")
print(pdf_path)
print(svg_path)
print(png_path)