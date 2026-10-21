# ============================================================
# Fig. 2c
# Dependence on annotated adduct information
#
# Four lines:
#   BUDDY
#   BUDDY (w/o Adduct)
#   FIDDLE
#   FIDDLE (w/o Adduct)
#
# Features:
# - BUDDY legend in the upper-left
# - FIDDLE legend in the lower-right
# - 12 value labels are all manually controllable
#
# Output:
#   fig2c_adduct_dependence.pdf
#   fig2c_adduct_dependence.svg
#   fig2c_adduct_dependence.png
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

    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "legend.fontsize": 6.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 150,
    "savefig.dpi": 600,
})


# ============================================================
# 2. Figure size
# ============================================================

FIG2C_WIDTH = 3.5
FIG2C_HEIGHT = 2.5


# ============================================================
# 3. Data
# ============================================================

x = np.arange(3)

xlabels = [
    "Top-1",
    "Top-3",
    "Top-5",
]

buddy = np.array([46.88, 56.96, 60.35])
buddy_wo_adduct = np.array([15.41, 42.49, 49.34])
fiddle = np.array([23.76, 34.94, 38.29])
fiddle_wo_adduct = np.array([10.89, 23.42, 28.68])


# ============================================================
# 4. Colors
# ============================================================

COLOR_BUDDY = "#6FA8DC"
COLOR_BUDDY_WO = "#9EC5E8"

COLOR_FIDDLE = "#73B66B"
COLOR_FIDDLE_WO = "#A6D39B"

GRID_COLOR = "#D9D9D9"


# ============================================================
# 5. Create figure
# ============================================================

fig, ax = plt.subplots(figsize=(FIG2C_WIDTH, FIG2C_HEIGHT))


# ============================================================
# 6. Plot lines
# ============================================================

line_buddy, = ax.plot(
    x, buddy,
    color=COLOR_BUDDY,
    linewidth=1.70,
    marker="s",
    markersize=6.2,
    markerfacecolor=COLOR_BUDDY,
    markeredgecolor="white",
    markeredgewidth=0.65,
    label="BUDDY",
    zorder=5,
)

line_buddy_wo, = ax.plot(
    x, buddy_wo_adduct,
    color=COLOR_BUDDY_WO,
    linewidth=1.50,
    marker="s",
    markersize=5.8,
    markerfacecolor=COLOR_BUDDY_WO,
    markeredgecolor="white",
    markeredgewidth=0.65,
    linestyle="--",
    label="BUDDY (w/o Adduct)",
    zorder=4,
)

line_fiddle, = ax.plot(
    x, fiddle,
    color=COLOR_FIDDLE,
    linewidth=1.70,
    marker="v",
    markersize=6.3,
    markerfacecolor=COLOR_FIDDLE,
    markeredgecolor="white",
    markeredgewidth=0.65,
    label="FIDDLE",
    zorder=5,
)

line_fiddle_wo, = ax.plot(
    x, fiddle_wo_adduct,
    color=COLOR_FIDDLE_WO,
    linewidth=1.50,
    marker="v",
    markersize=5.9,
    markerfacecolor=COLOR_FIDDLE_WO,
    markeredgecolor="white",
    markeredgewidth=0.65,
    linestyle="--",
    label="FIDDLE (w/o Adduct)",
    zorder=4,
)


# ============================================================
# 7. Axes
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(xlabels)

ax.set_ylabel(
    "Formula accuracy (%)",
    labelpad=5,
    fontweight="bold",
)

ax.set_ylim(8, 65)
ax.set_yticks([10, 20, 30, 40, 50, 60])

ax.grid(
    axis="y",
    linestyle=(0, (3, 3)),
    linewidth=0.65,
    color=GRID_COLOR,
)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 8. Two legends
# ============================================================

legend_buddy = ax.legend(
    handles=[line_buddy, line_buddy_wo],
    labels=["BUDDY", "BUDDY (w/o Adduct)"],
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    ncol=1,
    frameon=False,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.45,
    borderaxespad=0.0,
    fontsize=6.8,
)

for text in legend_buddy.get_texts():
    text.set_fontsize(6.8)
    text.set_fontweight("normal")

ax.add_artist(legend_buddy)

legend_fiddle = ax.legend(
    handles=[line_fiddle, line_fiddle_wo],
    labels=["FIDDLE", "FIDDLE (w/o Adduct)"],
    loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    ncol=1,
    frameon=False,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.45,
    borderaxespad=0.0,
    fontsize=6.8,
)

for text in legend_fiddle.get_texts():
    text.set_fontsize(6.8)
    text.set_fontweight("normal")


# ============================================================
# 9. MANUAL LABEL POSITIONS
# ============================================================
#
# You can manually control all 12 value labels here.
#
# Each tuple is:
#   (dx, dy, ha, va)
#
# Meaning:
#   dx: horizontal offset in points
#       > 0 moves right
#       < 0 moves left
#
#   dy: vertical offset in points
#       > 0 moves up
#       < 0 moves down
#
#   ha: horizontal alignment
#       "left", "center", "right"
#
#   va: vertical alignment
#       "bottom", "center", "top"
#
# If one label overlaps something, just adjust its dx/dy.
# ============================================================

buddy_label_pos = [
    (-8, 4, "left", "bottom"),    # 46.88, Top-1
    (-8, 4, "left", "bottom"),    # 56.96, Top-3
    (0, 3, "center", "bottom"),   # 60.35, Top-5
]

buddy_wo_label_pos = [
    (-9, 4, "left", "bottom"),   # 15.41, Top-1
    (-8, 4, "left", "bottom"),     # 42.49, Top-3
    (0, 3, "center", "bottom"),    # 49.34, Top-5
]

fiddle_label_pos = [
    (-8, 4, "left", "bottom"),    # 23.76, Top-1
    (0, 10, "center", "top"),     # 34.94, Top-3
    (0, 3, "center", "bottom"),    # 38.29, Top-5
]

fiddle_wo_label_pos = [
    (-8, -1, "left", "top"),       # 10.89, Top-1
    (0, -4, "center", "top"),     # 23.42, Top-3
    (0, -4, "center", "top"),     # 28.68, Top-5
]


# ============================================================
# 10. Helper for manual labels
# ============================================================

def add_manual_labels(ax, xs, ys, color, positions,
                      fontsize=7.0, fontweight="normal"):
    """
    Add labels with manually specified offsets.

    positions: list of tuples
        (dx, dy, ha, va)
    """
    for xi, yi, (dx, dy, ha, va) in zip(xs, ys, positions):
        ax.annotate(
            f"{yi:.2f}",
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            zorder=6,
            clip_on=True,
        )


# ============================================================
# 11. Draw all 12 labels
# ============================================================

add_manual_labels(
    ax, x, buddy, COLOR_BUDDY,
    buddy_label_pos,
    fontsize=7.2,
    fontweight="bold"
)

add_manual_labels(
    ax, x, buddy_wo_adduct, COLOR_BUDDY_WO,
    buddy_wo_label_pos,
    fontsize=7.0,
    fontweight="normal"
)

add_manual_labels(
    ax, x, fiddle, COLOR_FIDDLE,
    fiddle_label_pos,
    fontsize=7.2,
    fontweight="bold"
)

add_manual_labels(
    ax, x, fiddle_wo_adduct, COLOR_FIDDLE_WO,
    fiddle_wo_label_pos,
    fontsize=7.0,
    fontweight="normal"
)


# ============================================================
# 12. Layout
# ============================================================

plt.subplots_adjust(
    left=0.18,
    right=0.98,
    top=0.96,
    bottom=0.16,
)


# ============================================================
# 13. Save
# ============================================================

out_dir = Path("./figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "fig2c_adduct_dependence.pdf"
svg_path = out_dir / "fig2c_adduct_dependence.svg"
png_path = out_dir / "fig2c_adduct_dependence.png"

fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=600)

plt.show()

print("Saved files:")
print(pdf_path)
print(svg_path)
print(png_path)