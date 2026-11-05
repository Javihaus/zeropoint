"""Right decision against time per decision, for every system on the three tests no model is likely to have seen.

    python chart.py                      # reads results/summary.json, writes ../../assets/decision-models-chart.png

Drawn in the format of the earlier ZeroPoint charts: 2400 x 1600 px, transparent background, Archivo (OFL, in
fonts/), the same blues, zones and marker sizes. Every value comes from results/summary.json, written by
report.py; nothing is typed in by hand.

  x       median time per decision, on the hardware each system needs (log scale)
  y       right decision, none included, on the 900 cases
  fill    silent mistakes: an option picked with probability 0.9 or more when none is right (white: no
          probabilities)
  ring    a second ring: the answer changes when the same options come in another order; dashed: not measured
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
for w in ["400Regular", "500Medium", "600SemiBold"]:
    path = os.path.join(HERE, "fonts", w, f"Archivo_{w}.ttf")
    if os.path.exists(path):
        fm.fontManager.addfont(path)
plt.rcParams.update({"font.family": "Archivo"})

DPI = 120
px = lambda v: v * 72 / DPI                         # size in pixels -> points
BLUE, LINE, ORANGE, LIGHT = "#6a8bb5", "#86a3c9", "#e07f5f", "#a9c1e0"
ZONE_GREEN, ZONE_AMBER, ZONE_RED = (75 / 255, 174 / 255, 124 / 255, 0.16), (224 / 255, 168 / 255, 81 / 255, 0.16), \
    (214 / 255, 82 / 255, 107 / 255, 0.12)
TXT_GREEN, TXT_AMBER, TXT_RED = "#3f9c6c", "#c48a35", "#c9566a"
BIG = px(58) ** 2
SILENT = LinearSegmentedColormap.from_list("silent", ["#dfe8f4", "#6a8bb5", "#2b3f66"])
SILENT_MAX = 0.30

# system -> label, label position (x in ms, y), alignment, vertical alignment
LABELS = {
    "zeropoint": ("ZeroPoint", (42, 0.505), "center", "top"),
    "strands_gpu": ("Strands Decider 2B, GPU", (93, 0.585), "center", "bottom"),
    "strands_cpu": ("the same model, CPU", (6770, 0.585), "center", "bottom"),
    "clef_flash_none": ("Clef-flash, GPU", (156, 0.505), "center", "top"),
    "laya_none": ("Laya, CPU", (593, 0.47), "center", "top"),
    "semantic_router": ("Semantic Router, CPU", (24, 0.212), "left", "top"),
    "nli": ("Zero-shot NLI, CPU", (1697, 0.175), "center", "top"),
    "claude": ("Claude 5.5 · API", (1250, 0.603), "center", "bottom"),
    "gpt": ("GPT-6 and 6.1 · API", (2100, 0.555), "center", "top"),
}


def main(summary_path=os.path.join(HERE, "results", "summary.json"),
         out_path=os.path.join(HERE, "..", "..", "assets", "decision-models-chart.png")):
    S = json.load(open(summary_path, encoding="utf-8"))
    C = S["llm_comparison"]["systems"]
    tests = S["llm_comparison"]["tests"]

    def order(s):
        v = [S["systems"][s]["tests"].get(t, {}).get("order_changed") for t in tests]
        return None if any(x is None for x in v) else sum(v) / len(v)

    fig = plt.figure(figsize=(2400 / DPI, 1600 / DPI), dpi=DPI)
    ax = fig.add_axes([240 / 2400, 200 / 1600, 1880 / 2400, 1282 / 1600])
    ax.set_xscale("log")
    ax.set_xlim(10, 20000)
    ax.set_ylim(0.1, 0.7)
    ax.patch.set_alpha(0)

    # human response-time limits: under 0.1 s feels instant, under 1 s keeps the flow, over 1 s is a wait
    ax.add_patch(Rectangle((10, 0.1), 90, 0.6, color=ZONE_GREEN, lw=0, zorder=0))
    ax.add_patch(Rectangle((100, 0.1), 900, 0.6, color=ZONE_AMBER, lw=0, zorder=0))
    ax.add_patch(Rectangle((1000, 0.1), 19000, 0.6, color=ZONE_RED, lw=0, zorder=0))
    ax.grid(color=LINE, alpha=0.22, lw=px(2), zorder=0.5)
    for x in (100, 1000):
        ax.axvline(x, color=LINE, lw=px(2), dashes=(5, 5), zorder=1)
    ax.text(12, 0.11, "instant", fontsize=px(36), color=TXT_GREEN, weight="semibold")
    ax.text(115, 0.11, "a pause", fontsize=px(36), color=TXT_AMBER, weight="semibold")
    ax.text(17500, 0.11, "a wait", fontsize=px(36), color=TXT_RED, weight="semibold", ha="right")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BLUE)
        ax.spines[s].set_linewidth(px(4))
    ticks = [10, 30, 100, 300, 1000, 3000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["10 ms", "30 ms", "100 ms", "300 ms", "1 s", "3 s", "10 s"])
    ax.minorticks_off()
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%"])
    ax.tick_params(colors=BLUE, labelsize=px(34), length=0, pad=px(22))
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("medium")
    ax.set_xlabel("time per decision, median · on the hardware each one needs", fontsize=px(40), color=BLUE,
                  weight="medium", labelpad=px(36))
    ax.set_ylabel("right decision, none included", fontsize=px(40), color=BLUE, weight="medium", labelpad=px(40))
    fig.text(244 / 2400, 1 - 30 / 1600, "How often it is right, and how long it takes  ·  900 cases no model had seen",
             fontsize=px(52), color=BLUE, weight="semibold", va="top")

    def dot(x, y, silent, ring, zorder=4):
        fill = "white" if silent is None else SILENT(min(silent, SILENT_MAX) / SILENT_MAX)
        ax.scatter([x], [y], s=BIG, color=fill, edgecolor=BLUE, lw=px(3), zorder=zorder)
        if ring == "changes":
            ax.scatter([x], [y], s=(px(58) + px(26)) ** 2, facecolor="none", edgecolor=BLUE, lw=px(4),
                       zorder=zorder)
        elif ring == "not measured":
            ax.scatter([x], [y], s=(px(58) + px(26)) ** 2, facecolor="none", edgecolor=LINE, lw=px(3),
                       linestyle=(0, (2, 2)), zorder=zorder)

    def ring_of(s):
        if s.startswith("llm_"):
            return "not measured"
        o = order(s)
        return "changes" if o and o > 0 else None

    def label(key, sub, color=BLUE):
        text, (lx, ly), ha, va = LABELS[key]
        if va == "bottom":                     # above the point: name, then the figures just under it
            ax.text(lx, ly + 0.074, text, fontsize=px(41), color=color, weight="semibold", ha=ha, va="bottom",
                    zorder=5)
            ax.text(lx, ly + 0.042, sub, fontsize=px(33), color=color, ha=ha, va="bottom", zorder=5)
        else:                                  # under or beside the point
            ax.text(lx, ly, text, fontsize=px(41), color=color, weight="semibold", ha=ha, va="top", zorder=5)
            ax.text(lx, ly - 0.032, sub, fontsize=px(33), color=color, ha=ha, va="top", zorder=5)

    pct = lambda v: f"{100 * v:.1f}%"
    for s in ["zeropoint", "clef_flash_none", "laya_none", "semantic_router", "nli"]:
        r = C[s]
        dot(r["ms_median"], r["right"], r["silent"], ring_of(s))
        sub = f"{pct(r['right'])} · {r['ms_median']:,.0f} ms"
        label(s, sub, ORANGE if s == "zeropoint" else BLUE)

    st = C["strands_none"]
    cpu = S.get("cpu_latency", {}).get("strands_none", {}).get("median")
    dot(st["ms_median"], st["right"], st["silent"], ring_of("strands_none"))
    label("strands_gpu", f"{pct(st['right'])} · {st['ms_median']:,.0f} ms")
    if cpu:
        ax.plot([st["ms_median"] * 1.25, cpu / 1.25], [st["right"], st["right"]], color=LINE, lw=px(3),
                dashes=(4, 4), alpha=0.6, zorder=2)
        dot(cpu, st["right"], st["silent"], ring_of("strands_none"), zorder=3)
        label("strands_cpu", f"{cpu / 1000:.1f} s")

    for group, key in [("llm_claude", "claude"), ("llm_gpt", "gpt")]:
        members = [s for s in C if s.startswith(group)]
        for s in members:
            dot(C[s]["ms_median"], C[s]["right"], C[s]["silent"], ring_of(s))
        lo = min(C[s]["right"] for s in members)
        hi = max(C[s]["right"] for s in members)
        t0 = min(C[s]["ms_median"] for s in members)
        t1 = max(C[s]["ms_median"] for s in members)
        label(key, f"{pct(lo)}–{pct(hi)} · {t0 / 1000:.1f}–{t1 / 1000:.1f} s")

    # legend: fill = silent mistakes, rings = order
    cax = fig.add_axes([2150 / 2400, 560 / 1600, 28 / 2400, 760 / 1600])
    grad = [[i / 255] for i in range(256)]
    cax.imshow(grad, aspect="auto", cmap=SILENT, origin="lower", extent=(0, 1, 0, SILENT_MAX))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 0.1, 0.2, 0.3])
    cax.set_yticklabels(["0%", "10%", "20%", "30%"])
    cax.tick_params(colors=BLUE, labelsize=px(30), length=0, pad=px(12))
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text(2164 / 2400, (560 + 760 + 24) / 1600, "silent\nmistakes", fontsize=px(30), color=BLUE,
             weight="semibold", ha="center", va="bottom", linespacing=1.1)

    lx, ly = 13, 0.40
    dot(lx, ly, 0.0, None, zorder=6)
    ax.text(lx * 1.35, ly, "same answer whatever the order", fontsize=px(31.6), color=LINE, va="center", zorder=6)
    dot(lx, ly - 0.045, 0.0, "changes", zorder=6)
    ax.text(lx * 1.35, ly - 0.045, "changes with the order", fontsize=px(31.6), color=LINE, va="center", zorder=6)
    dot(lx, ly - 0.09, 0.0, "not measured", zorder=6)
    ax.text(lx * 1.35, ly - 0.09, "order not measured", fontsize=px(31.6), color=LINE, va="center", zorder=6)

    fig.savefig(out_path, dpi=DPI, transparent=True)


if __name__ == "__main__":
    main()
