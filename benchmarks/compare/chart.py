"""Right decision against latency on a CPU, for every system on the three tests no model is likely to have seen.

    python chart.py                      # reads results/summary.json, writes ../../assets/decision-models-chart.png

Drawn in the format of the earlier ZeroPoint charts: 2400 x 1600 px, Archivo (OFL, in fonts/), the same marker
sizes; white background, two axes and one colour scale. Every value comes from results/summary.json, written by
report.py; nothing is typed in by hand.

  x       median latency on a CPU (log scale); the LLMs are called from a CPU, network included; Strands Decider
          from cpu_latency.py; Clef-flash needs a GPU and is left out
  y       right decision, none included, on the 900 cases
  colour  silent mistakes: an option picked with probability 0.9 or more when none is right (white: no
          probabilities)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap


def _json_load(*args, **kw):
    with open(*args, **kw) as f:
        return json.load(f)


def _jsonl(*args, **kw):
    with open(*args, **kw) as f:
        return [json.loads(line) for line in f]

HERE = os.path.dirname(os.path.abspath(__file__))
for w in ["400Regular", "500Medium", "600SemiBold"]:
    path = os.path.join(HERE, "fonts", w, f"Archivo_{w}.ttf")
    if os.path.exists(path):
        fm.fontManager.addfont(path)
plt.rcParams.update({"font.family": "Archivo"})

DPI = 120
px = lambda v: v * 72 / DPI                         # size in pixels -> points
BLUE, LINE, ORANGE, LIGHT = "#1f4a85", "#3d679f", "#d9622f", "#a9c1e0"
BIG = px(58) ** 2
SILENT = LinearSegmentedColormap.from_list("silent", ["#fdeee4", "#e8894f", "#9c3412"])
SILENT_MAX = 0.30

# system -> label, label position (x in ms, y), alignment, vertical alignment
LABELS = {
    "zeropoint": ("ZeroPoint", (42, 0.505), "center", "top"),
    "strands_cpu": ("Strands Decider 2B ¹", (6770, 0.585), "center", "bottom"),
    "laya_none": ("Laya", (593, 0.47), "center", "top"),
    "semantic_router": ("Semantic Router", (24, 0.212), "left", "top"),
    "nli": ("Zero-shot NLI", (1697, 0.175), "center", "top"),
    "claude": ("Claude 5.5 · API", (1250, 0.603), "center", "bottom"),
    "gpt": ("GPT-6 and 6.1 · API", (2100, 0.555), "center", "top"),
}


def main(summary_path=os.path.join(HERE, "results", "summary.json"),
         out_path=os.path.join(HERE, "..", "..", "assets", "decision-models-chart.png")):
    S = _json_load(summary_path, encoding="utf-8")
    C = S["llm_comparison"]["systems"]

    fig = plt.figure(figsize=(2400 / DPI, 1600 / DPI), dpi=DPI)
    ax = fig.add_axes([250 / 2400, 285 / 1600, 1790 / 2400, 1125 / 1600])
    ax.set_xscale("log")
    ax.set_xlim(10, 20000)
    ax.set_ylim(0.1, 0.7)
    ax.set_facecolor("white")

    ax.grid(color=LINE, alpha=0.18, lw=px(2), zorder=0.5)

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
    ax.tick_params(colors=BLUE, labelsize=px(40), length=0, pad=px(22))
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("medium")
    ax.set_xlabel("Latency in CPU", fontsize=px(47), color=BLUE, weight="medium", labelpad=px(30))
    ax.set_ylabel("Accuracy", fontsize=px(47), color=BLUE, weight="medium", labelpad=px(34))
    fig.text(244 / 2400, 1 - 30 / 1600, "How often it is right, and how long it takes on a CPU  ·  900 new cases",
             fontsize=px(52), color=BLUE, weight="semibold", va="top")

    def dot(x, y, silent, zorder=4):
        fill = "white" if silent is None else SILENT(min(silent, SILENT_MAX) / SILENT_MAX)
        ax.scatter([x], [y], s=BIG, color=fill, edgecolor=BLUE, lw=px(3), zorder=zorder)

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
    for s in ["zeropoint", "laya_none", "semantic_router", "nli"]:
        r = C[s]
        dot(r["ms_median"], r["right"], r["silent"])
        sub = f"{pct(r['right'])} · {r['ms_median']:,.0f} ms"
        label(s, sub, ORANGE if s == "zeropoint" else BLUE)

    st = C["strands_none"]
    cpu = S.get("cpu_latency", {}).get("strands_none", {}).get("median")
    if cpu:
        dot(cpu, st["right"], st["silent"])
        label("strands_cpu", f"{pct(st['right'])} · {cpu / 1000:.1f} s")

    for group, key in [("llm_claude", "claude"), ("llm_gpt", "gpt")]:
        members = [s for s in C if s.startswith(group)]
        for s in members:
            dot(C[s]["ms_median"], C[s]["right"], C[s]["silent"])
        lo = min(C[s]["right"] for s in members)
        hi = max(C[s]["right"] for s in members)
        t0 = min(C[s]["ms_median"] for s in members)
        t1 = max(C[s]["ms_median"] for s in members)
        label(key, f"{pct(lo)}–{pct(hi)} · {t0 / 1000:.1f}–{t1 / 1000:.1f} s")

    # colour scale: silent mistakes
    cax = fig.add_axes([2215 / 2400, 520 / 1600, 30 / 2400, 720 / 1600])
    grad = [[i / 255] for i in range(256)]
    cax.imshow(grad, aspect="auto", cmap=SILENT, origin="lower", extent=(0, 1, 0, SILENT_MAX))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 0.1, 0.2, 0.3])
    cax.set_yticklabels(["0%", "10%", "20%", "30%"])
    cax.tick_params(colors=BLUE, labelsize=px(35), length=0, pad=px(12))
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text(2230 / 2400, (520 + 720 + 28) / 1600, "silent\nmistakes", fontsize=px(35), color=BLUE,
             weight="semibold", ha="center", va="bottom", linespacing=1.1)

    L = S.get("cpu_latency", {})
    zp_same = L.get("zeropoint", {}).get("median")
    note = ("¹ " + f"on another 2-core CPU, {L.get('cases', 0)} cases of the 2,000-case test, where ZeroPoint takes "
            f"{zp_same:.0f} ms" if zp_same else "")
    fig.text(244 / 2400, 22 / 1600, note + ".\nClef-flash needs a GPU: not measured on a CPU.  API: called from a "
             "CPU, network included.", fontsize=px(28), color=LINE, va="bottom", linespacing=1.4)
    fig.savefig(out_path, dpi=DPI, facecolor="white")


if __name__ == "__main__":
    main()
