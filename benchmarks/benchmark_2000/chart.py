"""AUROC against FPR at 95% recall on the 2,000-case benchmark.

Drawn to match assets/benchmark-chart.png exactly: 2400 x 1600 px, transparent background, Archivo, the same
zones, colours, marker sizes and line styles (all measured from that file).

    python chart.py results.json ../../assets/benchmark-2000-chart.png

Needs the Archivo font (OFL, Google Fonts): put the .ttf files in fonts/<weight>/Archivo_<weight>.ttf next to
this script, or point ARCHIVO_DIR at them. Without it, matplotlib falls back to its default font.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.environ.get("ARCHIVO_DIR", os.path.join(HERE, "fonts"))
for w in ["400Regular", "500Medium", "600SemiBold"]:
    path = os.path.join(FONTS, w, f"Archivo_{w}.ttf")
    if os.path.exists(path):
        fm.fontManager.addfont(path)
plt.rcParams.update({"font.family": "Archivo", "mathtext.fontset": "stix"})

DPI = 120
PT = 72 / DPI                     # points per pixel
px = lambda v: v * PT             # size in pixels -> points

BLUE, LINE, ORANGE, LIGHT = "#6a8bb5", "#86a3c9", "#e07f5f", "#a9c1e0"
ZONE_GREEN, ZONE_AMBER, ZONE_RED = (75 / 255, 174 / 255, 124 / 255, 0.16), (224 / 255, 168 / 255, 81 / 255, 0.16), \
    (214 / 255, 82 / 255, 107 / 255, 0.12)
TXT_GREEN, TXT_AMBER, TXT_RED = "#3f9c6c", "#c48a35", "#c9566a"

BIG, SMALL = px(58) ** 2, px(28) ** 2          # marker areas (pt^2) for 58 px and 28 px circles

# system -> label, colour, filled, label anchor (x, y in data units), alignment
STYLE = {
    "zeropoint": ("ZeroPoint", ORANGE, True, (0.395, 0.857), "right"),
    "zeropoint_k025": (r"ZeroPoint, $\kappa$ = 0.25", ORANGE, False, (0.455, 0.80), "right"),
    "sr_fewshot": ("Semantic Router, 10 examples", LINE, False, (0.525, 0.905), "left"),
    "sr_names": ("Semantic Router, names only", LIGHT, False, (1.0, 0.86), "right"),
    "laya": ("Laya", BLUE, True, (1.0, 0.745), "right"),
    "nli_deberta": ("Zero-shot NLI (DeBERTa)", LIGHT, True, (0.755, 0.79), "right"),
}


def main(results_path, out_path):
    R = json.load(open(results_path))
    fig = plt.figure(figsize=(2400 / DPI, 1600 / DPI), dpi=DPI)
    ax = fig.add_axes([240 / 2400, 200 / 1600, 2000 / 2400, 1282 / 1600])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0203)
    ax.patch.set_alpha(0)

    ax.add_patch(Rectangle((0, 0.7), 0.5, 0.3203, color=ZONE_GREEN, lw=0, zorder=0))
    ax.add_patch(Rectangle((0.5, 0.7), 0.5, 0.3203, color=ZONE_AMBER, lw=0, zorder=0))
    ax.add_patch(Rectangle((0, 0.4), 1, 0.3, color=ZONE_RED, lw=0, zorder=0))
    ax.grid(color=LINE, alpha=0.22, lw=px(2), zorder=0.5)
    ax.axhline(0.7, color=LINE, lw=px(2), dashes=(5, 5), zorder=1)
    ax.plot([0.5, 0.5], [0.7, 1.0203], color=LINE, lw=px(2), dashes=(5, 5), zorder=1)
    ax.axhline(0.5, color=LINE, lw=px(3), dashes=(1.33, 3.33), alpha=0.6, zorder=1)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BLUE)
        ax.spines[s].set_linewidth(px(4))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.tick_params(colors=BLUE, labelsize=px(34), length=0, pad=px(22))
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("medium")
    ax.set_xlabel("FPR at 95% recall · lower is better", fontsize=px(40), color=BLUE, weight="medium",
                  labelpad=px(36))
    ax.set_ylabel("AUROC · higher is better", fontsize=px(40), color=BLUE, weight="medium", labelpad=px(40))
    fig.text(244 / 2400, 1 - 30 / 1600, "Does it know when none of the options is right?  2,000 new cases",
             fontsize=px(52), color=BLUE, weight="semibold", va="top")

    ax.text(0.015, 0.712, "knows when none fits", fontsize=px(36), color=TXT_GREEN, weight="semibold")
    ax.text(0.99, 0.99, "ranks cases, but acts on many\nwith no right answer", fontsize=px(36), color=TXT_AMBER,
            weight="semibold", ha="right", va="top", linespacing=1.15)
    ax.text(0.99, 0.416, "cannot tell", fontsize=px(36), color=TXT_RED, weight="semibold", ha="right")
    ax.text(0.015, 0.512, "chance", fontsize=px(35), color=LINE)

    for n, (label, col, filled, (lx, ly), ha) in STYLE.items():
        if n not in R:
            continue
        for part in ("everyday", "support"):
            e = R[n][part][0]
            if filled:
                ax.scatter([e["fpr95"]], [e["auroc"]], s=SMALL, color=col, alpha=0.79, lw=0, zorder=3)
            else:
                ax.scatter([e["fpr95"]], [e["auroc"]], s=SMALL, facecolor="none", edgecolor=col, lw=px(4), zorder=3)
        e = R[n]["all"][0]
        if filled:
            ax.scatter([e["fpr95"]], [e["auroc"]], s=BIG, color=col, lw=0, zorder=4)
        else:
            ax.scatter([e["fpr95"]], [e["auroc"]], s=BIG, facecolor="none", edgecolor=col, lw=px(6), zorder=4)
        tcol = LINE if col == LIGHT else col
        ax.text(lx, ly, label, fontsize=px(43.5), color=tcol, weight="semibold", ha=ha, va="bottom", zorder=5)
        ax.text(lx, ly - 0.006, f"AUROC {e['auroc']:.2f} · FPR {e['fpr95']:.2f}", fontsize=px(36), color=tcol,
                ha=ha, va="top", zorder=5)

    # legend, as in the original: big marker = the whole benchmark, small = one part of it
    ax.scatter([0.0175], [0.4415], s=px(36) ** 2, color=LINE, lw=0, zorder=4)
    ax.text(0.037, 0.4415, "all 2,000 cases", fontsize=px(31.6), color=LINE, va="center")
    ax.scatter([0.21], [0.4415], s=px(20) ** 2, color=LINE, alpha=0.78, lw=0, zorder=4)
    ax.text(0.225, 0.4415, "each landscape (everyday, support)", fontsize=px(31.6), color=LINE, va="center")

    fig.savefig(out_path, dpi=DPI, transparent=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.json"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "benchmark-2000-chart.png"))
