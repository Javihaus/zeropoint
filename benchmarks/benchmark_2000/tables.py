"""Markdown tables from results.json and out/*.json: overall, per landscape, per scenario."""

import json
import os

import numpy as np

from metrics import DATA, NAMES, threshold_at_tpr

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
ORDER = [n for n in NAMES if n in R]
f3 = lambda x: f"{x:.3f}"


def ci(e, c, k):
    return f"{e[k]:.3f} [{c[k][0]:.2f}, {c[k][1]:.2f}]"


lines = ["## Overall (2,000 cases)", "",
         "| System | Accuracy, choosing | Accuracy, default | Accuracy @ TPR95 | AUROC | FPR95 |",
         "|---|---|---|---|---|---|"]
for n in ORDER:
    e, c = R[n]["all"]
    lines.append(f"| {NAMES[n]} | {ci(e, c, 'acc_choice')} | {f3(e['acc_default'])} | {f3(e['acc_tpr95'])} | "
                 f"{ci(e, c, 'auroc')} | {ci(e, c, 'fpr95')} |")

for part, title in [("everyday", "everyday (1,415 cases)"), ("support", "support (585 cases)"),
                    ("no_clinc_overlap", "without the 23 cases whose text is in CLINC150")]:
    lines += ["", f"## {title}", "", "| System | Accuracy, choosing | AUROC | FPR95 |", "|---|---|---|---|"]
    for n in ORDER:
        e, c = R[n][part]
        lines.append(f"| {NAMES[n]} | {f3(e['acc_choice'])} | {ci(e, c, 'auroc')} | {ci(e, c, 'fpr95')} |")

# per scenario, at each system's own TPR95 threshold (set on all 1,000 answerable cases)
scen_pos = ["In-Scope", "Boundary-Answerable", "Adversarial-Edge"]
scen_neg = ["Out-of-Scope", "Boundary-NearMiss", "OOD-Noise"]
lines += ["", "## Per scenario, at the threshold that keeps 95% of answerable cases", "",
          "Answerable scenarios: share acted on with the right option. No-answer scenarios: share correctly "
          "answered *none*.", "",
          "| System | " + " | ".join(scen_pos + scen_neg) + " |", "|---" * (1 + len(scen_pos + scen_neg)) + "|"]
for n in ORDER:
    rows = json.load(open(os.path.join(HERE, "out", f"{n}.json")))
    conf = np.array([r["conf"] for r in rows])
    pos = np.array([DATA[r["id"]]["expected_choice"] is not None for r in rows])
    th = threshold_at_tpr(conf[pos])
    cells = []
    for s in scen_pos + scen_neg:
        v = []
        for r in rows:
            c = DATA[r["id"]]
            if c["scenario"] != s:
                continue
            act = r["conf"] >= th
            if c["expected_choice"] is None:
                v.append(not act)
            else:
                v.append(act and max(r["scores"], key=r["scores"].get) == c["expected_choice"])
        cells.append(f"{np.mean(v):.2f}")
    lines.append(f"| {NAMES[n]} | " + " | ".join(cells) + " |")

open(os.path.join(HERE, "tables.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
