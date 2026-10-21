# ============================================================
# Fig. 2e
# Reference-formula coverage
#
# Compact horizontal 100% stacked bars
#
# Designed for the limited space available in Fig. 2e
#
# Output:
#   fig2e_reference_formula_coverage.pdf
#   fig2e_reference_formula_coverage.svg
#   fig2e_reference_formula_coverage.png
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch


# ============================================================
# 1. Global style
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,

    "axes.labelsize": 7.2,
    "axes.linewidth": 0.8,

    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.major.size": 3,
    "ytick.major.size": 0,

    "legend.fontsize": 6.3,

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
# Wide and short, specifically for the available Fig. 2e space.
# ============================================================

FIG_WIDTH = 2.2
FIG_HEIGHT = 1.55


# ============================================================
# 3. Data
# ============================================================

covered = np.array([
    76.03,
    95.63,
])

missed = 100.0 - covered

improvement = covered[1] - covered[0]


# ============================================================
# 4. Colors
# ============================================================

COVERED_COLOR = "#6F7082"
MISSED_COLOR = "#E3E3E6"

TEXT_COLOR = "#202020"
SECONDARY_TEXT = "#555555"

# Same red family already used elsewhere in Fig. 2
IMPROVEMENT_COLOR = "#C92A2A"


# ============================================================
# 5. Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        FIG_WIDTH,
        FIG_HEIGHT,
    )
)


# ============================================================
# 6. Bar geometry
# ============================================================
#
# Leave vertical space above each bar for the category label.
# ============================================================

y_positions = [
    1.00,
    0.20,
]

BAR_HEIGHT = 0.25


# ============================================================
# 7. Draw stacked bars
# ============================================================

for y, cov, mis in zip(
    y_positions,
    covered,
    missed,
):

    # Covered
    ax.barh(
        y,
        cov,
        height=BAR_HEIGHT,
        left=0,

        color=COVERED_COLOR,
        edgecolor="none",

        zorder=3,
    )

    # Missed
    ax.barh(
        y,
        mis,
        height=BAR_HEIGHT,
        left=cov,

        color=MISSED_COLOR,
        edgecolor="none",

        zorder=2,
    )


# ============================================================
# 8. Category labels
# ============================================================
#
# Put labels ABOVE the bars instead of on the y-axis.
# This saves a lot of horizontal space.
# ============================================================

ax.text(
    0,
    y_positions[0] + 0.22,

    "Compact candidate pool",

    ha="left",
    va="bottom",

    fontsize=6,
    color=TEXT_COLOR,

    zorder=5,
)

ax.text(
    0,
    y_positions[1] + 0.22,

    "Chemically constrained space",

    ha="left",
    va="bottom",

    fontsize=6,
    color=TEXT_COLOR,

    zorder=5,
)


# ============================================================
# 9. Coverage values
# ============================================================
#
# Put the main coverage numbers inside the dark segments.
# ============================================================

ax.text(
    covered[0] - 2.0,
    y_positions[0],

    f"{covered[0]:.2f}%",

    ha="right",
    va="center",

    fontsize=6.8,
    fontweight="bold",

    color="white",

    zorder=6,
)

ax.text(
    covered[1] - 2.0,
    y_positions[1],

    f"{covered[1]:.2f}%",

    ha="right",
    va="center",

    fontsize=6.8,
    fontweight="bold",

    color="white",

    zorder=6,
)


# ============================================================
# 10. Improvement annotation
# ============================================================
#
# Small annotation emphasizing expansion of the coverage ceiling.
# ============================================================

ax.text(
    99,
    0.57,

    f"+{improvement:.2f} pp",

    ha="right",
    va="center",

    fontsize=6.5,
    fontweight="bold",

    color=IMPROVEMENT_COLOR,

    zorder=7,
)


# ============================================================
# 11. Axes
# ============================================================

ax.set_xlim(
    0,
    100,
)

ax.set_ylim(
    -0.12,
    1.48,
)


# Only three x ticks to keep the panel compact
ax.set_xticks([
    0,
    50,
    100,
])

# ax.set_xlabel(
#     "Coverage (%)",
#     fontsize=7.0,
#     fontweight="normal",
#     labelpad=2.5,
# )


# No y-axis needed
ax.set_yticks([])


# ============================================================
# 12. Grid
# ============================================================

ax.grid(
    axis="x",
    linestyle=(0, (3, 3)),
    linewidth=0.60,
    color="#D9D9D9",
)

ax.set_axisbelow(True)


# ============================================================
# 13. Clean frame
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.spines["bottom"].set_linewidth(0.8)


# ============================================================
# 14. Legend
# ============================================================

legend_handles = [
    Patch(
        facecolor=COVERED_COLOR,
        edgecolor="none",
        label="Formula covered",
    ),

    Patch(
        facecolor=MISSED_COLOR,
        edgecolor="none",
        label="Formula missed",
    ),
]


legend = ax.legend(
    handles=legend_handles,

    loc="upper center",

    bbox_to_anchor=(
        0.50,
        1.18,
    ),

    ncol=2,

    frameon=False,

    columnspacing=1.15,

    handlelength=1.35,
    handleheight=0.75,

    handletextpad=0.40,

    borderaxespad=0,

    fontsize=6.1,
)


for text in legend.get_texts():

    text.set_fontsize(6.1)
    text.set_fontweight("normal")


# ============================================================
# 15. Layout
# ============================================================
#
# Very little left margin because labels are now inside the axes.
# ============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.985,

    top=0.78,
    bottom=0.25,
)


# ============================================================
# 16. Save
# ============================================================

out_dir = Path("./figures")

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


pdf_path = (
    out_dir /
    "fig2e_reference_formula_coverage.pdf"
)

svg_path = (
    out_dir /
    "fig2e_reference_formula_coverage.svg"
)

png_path = (
    out_dir /
    "fig2e_reference_formula_coverage.png"
)


fig.savefig(
    pdf_path,

    bbox_inches="tight",

    pad_inches=0.03,

    facecolor="white",
)


fig.savefig(
    svg_path,

    bbox_inches="tight",

    pad_inches=0.03,

    facecolor="white",
)


fig.savefig(
    png_path,

    bbox_inches="tight",

    pad_inches=0.03,

    facecolor="white",

    dpi=600,
)


plt.show()


print("Saved files:")
print(pdf_path)
print(svg_path)
print(png_path)