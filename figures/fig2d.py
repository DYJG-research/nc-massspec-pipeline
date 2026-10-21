# ============================================================
# Fig. 2e
# Chemical reliability of molecular formula predictions
#
# Three compact paired-bar small multiples:
#   1. Mass-valid
#   2. Formula validity
#   3. Adduct accuracy
#
# Style follows the provided reference figure:
#   DeepSeek-V4-Pro -> light beige
#   Our Model       -> muted slate
#
# Each panel has its own visible axes and local y-range.
#
# Output:
#   fig2e_prediction_reliability.pdf
#   fig2e_prediction_reliability.svg
#   fig2e_prediction_reliability.png
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

    "axes.labelsize": 7.6,
    "axes.linewidth": 0.85,

    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "legend.fontsize": 6.8,

    # Editable text in vector output
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
# Deliberately shorter than previous version.
# This gives the compact appearance of the reference figure.
# ============================================================

FIG2E_WIDTH = 3.5
FIG2E_HEIGHT = 1.85


# ============================================================
# 3. Data
# ============================================================

metrics = [
    "Mass-valid",
    "Formula validity",
    "Adduct accuracy",
]

deepseek_values = np.array([
    91.25,
    100.00,
    79.90,
])

ours_values = np.array([
    99.99,
    99.99,
    70.86,
])


# ============================================================
# 4. Colors
# ============================================================
#
# Closely follow the supplied reference figure:
#
# Annotation without MS/MS -> light beige
# Annotation with MS/MS    -> grey/slate
#
# Here:
# DeepSeek-V4-Pro -> beige
# Our Model       -> slate
# ============================================================

COLOR_DEEPSEEK = "#E7C9A9"
COLOR_OURS = "#6F7082"

TEXT_COLOR = "#202020"


# ============================================================
# 5. Per-panel axis settings
# ============================================================
#
# Local y-ranges intentionally mimic the reference figure.
# This keeps each pair of bars compact.
#
# The truncated ranges are explicitly visible from the ticks.
# ============================================================

ylims = [
    (88, 101),    # Mass-valid
    (99.9, 100.02),    # Formula validity
    (68, 82),     # Adduct accuracy
]

yticks = [
    [90, 95, 100],
    [99.9, 99.95, 100],
    [70, 75, 80],
]


# ============================================================
# 6. Create figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(FIG2E_WIDTH, FIG2E_HEIGHT),
    sharey=False,
)


# ============================================================
# 7. Bar geometry
# ============================================================

BAR_WIDTH = 0.25

x_ds = -0.16
x_ours = 0.16


# ============================================================
# 8. Draw panels
# ============================================================

for i, ax in enumerate(axes):

    ds_value = deepseek_values[i]
    ours_value = ours_values[i]

    ymin, ymax = ylims[i]

    # --------------------------------------------------------
    # DeepSeek-V4-Pro
    # --------------------------------------------------------

    ax.bar(
        x_ds,
        ds_value,
        width=BAR_WIDTH,

        color=COLOR_DEEPSEEK,

        edgecolor="none",

        zorder=3,
    )


    # --------------------------------------------------------
    # Our Model
    # --------------------------------------------------------

    ax.bar(
        x_ours,
        ours_value,
        width=BAR_WIDTH,

        color=COLOR_OURS,

        edgecolor="none",

        zorder=3,
    )


    # --------------------------------------------------------
    # Numerical labels
    # --------------------------------------------------------
    #
    # Position relative to local axis height rather than fixed
    # absolute percentage points.
    # --------------------------------------------------------

    label_gap = (ymax - ymin) * 0.035

    ax.text(
        x_ds,
        ds_value + label_gap,

        f"{ds_value:.2f}",

        ha="center",
        va="bottom",

        fontsize=6.5,
        color=TEXT_COLOR,

        fontweight="normal",

        zorder=5,
    )

    ax.text(
        x_ours,
        ours_value + label_gap,

        f"{ours_value:.2f}",

        ha="center",
        va="bottom",

        fontsize=6.5,
        color=TEXT_COLOR,

        zorder=5,
    )


    # --------------------------------------------------------
    # Metric label below each subplot
    # --------------------------------------------------------

    ax.text(
        0.5,
        -0.08,
        metrics[i],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="normal",
    )


    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------
    #
    # No repeated model names.
    # Method identity is encoded by the shared legend.
    # --------------------------------------------------------

    ax.set_xlim(-0.46, 0.46)

    ax.set_xticks([])

    ax.tick_params(
        axis="x",
        bottom=True,
        labelbottom=False,
    )


    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_yticks(
        yticks[i]
    )

    if i == 0:
        ax.set_ylabel(
            "Rate (%)",
            fontsize=7.2,
            labelpad=2.5,
            fontweight="normal",
        )


    # --------------------------------------------------------
    # Axes style
    # --------------------------------------------------------
    #
    # Every panel keeps its own left and bottom axes,
    # matching the provided reference.
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.tick_params(
        axis="y",
        direction="out",
        pad=2,
    )

    # No grid lines, consistent with reference
    ax.grid(False)


# ============================================================
# 9. Shared legend
# ============================================================

legend_handles = [
    Patch(
        facecolor=COLOR_DEEPSEEK,
        edgecolor="none",
        label="Our Model",
    ),

    Patch(
        facecolor=COLOR_OURS,
        edgecolor="none",
        label="Our Model (w/o XGB + LLM)",
    ),
]


legend = fig.legend(
    handles=legend_handles,

    loc="upper center",

    # Higher than the three panels
    bbox_to_anchor=(0.5, 0.85),

    ncol=2,

    frameon=False,

    columnspacing=1.35,

    handlelength=1.45,
    handleheight=0.75,

    handletextpad=0.45,

    fontsize=6.8,
)


for text in legend.get_texts():

    text.set_fontsize(6.8)
    text.set_fontweight("normal")


# ============================================================
# 10. Layout
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.99,

    # Leave compact space under legend
    top=0.70,

    bottom=0.16,

    # Give the three panels visible separation
    wspace=0.60,
)


# ============================================================
# 11. Save
# ============================================================

out_dir = Path("./figures")

out_dir.mkdir(
    parents=True,
    exist_ok=True
)


pdf_path = (
    out_dir /
    "fig2e_prediction_reliability.pdf"
)

svg_path = (
    out_dir /
    "fig2e_prediction_reliability.svg"
)

png_path = (
    out_dir /
    "fig2e_prediction_reliability.png"
)


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