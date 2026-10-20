"""Figure for the round-3 event study (reads results/round3_events.json).

Usage: python scripts/round3_figures.py ../results/round3_events.json ../analysis/figures
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, outdir = sys.argv[1:3]
r = json.load(open(src))
SERIES = ["#2a78d6", "#eb6834"]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e2dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})

rows = [("lys_theta0_smoothed", "Exact LYS, neighbour-averaged"), ("lys_theta0", "Exact LYS map"),
        ("lys_theta0+epot", "Exact LYS + energy"), ("epot", "Per-atom energy"),
        ("epot_within_type", "Energy within particle type"), ("gnn_soft", "Soft modes + GNN"),
        ("gbr", "Soft modes + boosting"), ("type_only", "Particle type only"),
        ("softmode_raw", "Soft-mode estimate alone"), ("random", "Random")]
locators = [("atom", "Event at largest-displacement atom"),
            ("centre", "Event at neighbourhood-averaged centre")]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for j, ((key, label), c) in enumerate(zip(locators, SERIES)):
    off = -0.15 + 0.3 * j
    for i, (name, _) in enumerate(rows):
        m, (lo, hi) = r[name][key]["mean_percentile"], r[name][key]["ci95"]
        ax.plot([lo, hi], [i + off, i + off], color=c, lw=2, solid_capstyle="round")
        ax.plot(m, i + off, "o", color=c, ms=7, mec="white", mew=1.5,
                label=label if i == 0 else None)
ax.set_yticks(range(len(rows)), [l for _, l in rows])
ax.invert_yaxis()
ax.set_xlim(0.15, 0.75)
ax.axvline(0.5, color=GRID, lw=1)
ax.set_xlabel("Mean weakness percentile of the event (0 = predicted weakest)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_title(f"{r['n_events']} early plastic events, two ways of locating them (95% CI)",
             color=INK, loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(f"{outdir}/round3_events.png", dpi=150)
print("saved")
