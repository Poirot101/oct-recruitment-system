"""Figures for the round-2 study (reads results/round2.json).

Usage: python scripts/round2_figures.py ../results/round2.json ../analysis/figures
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, outdir = sys.argv[1:3]
r = json.load(open(src))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e2dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})

# data scaling: Spearman (min over directions) on the fixed 6-glass test set
fig, ax = plt.subplots(figsize=(6.4, 3.8))
names = [("gbr", "Soft modes + boosting"), ("gnn_soft", "Soft modes + GNN"),
         ("softmode_raw", "Soft-mode estimate alone"), ("gnn_raw", "Plain GNN")]
n = [row["n_train_glasses"] for row in r["scaling"]]
for (key, label), c in zip(names, SERIES):
    v = [row[key]["spearman_min_theta"] for row in r["scaling"]]
    ax.plot(n, v, color=c, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=label)
    dy = {"gnn_raw": 0.015, "softmode_raw": -0.015}.get(key, 0)
    ax.text(n[-1] + 0.3, v[-1] + dy, label, color=INK2, va="center", fontsize=9)
ax.set_xlabel("Training glasses (1024 sites each)")
ax.set_ylabel("Spearman vs exact map")
ax.set_xticks(n)
ax.set_xlim(1, 22)
ax.set_ylim(-0.05, 0.6)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_title("More data barely helps the hybrids; the plain GNN starts learning at 13 glasses",
             color=INK, loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(0.98, 0.05))
fig.tight_layout()
fig.savefig(f"{outdir}/round2_scaling.png", dpi=150)

# plastic events: mean percentile of the event atom in each weakness field
ev = r["events"]
rows = [("lys_exact", "Exact LYS map"), ("epot", "Per-atom energy"),
        ("gnn_soft", "Soft modes + GNN"), ("gbr", "Soft modes + boosting"),
        ("softmode_raw", "Soft-mode estimate alone"), ("random", "Random")]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for i, (key, label) in enumerate(rows):
    m, (lo, hi) = ev[key]["mean_percentile"], ev[key]["ci95"]
    ax.plot([lo, hi], [i, i], color=SERIES[0], lw=2, solid_capstyle="round")
    ax.plot(m, i, "o", color=SERIES[0], ms=8, mec="white", mew=1.5)
    ax.text(hi + 0.01, i, f"{m:.2f}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(rows)), [l for _, l in rows])
ax.invert_yaxis()
ax.set_xlim(0, 0.7)
ax.axvline(0.5, color=GRID, lw=1)
ax.set_xlabel("Weakness percentile of the yielding atom (0 = weakest)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_title(f"First 10 plastic events in {ev['n_glasses']} glasses (95% bootstrap CI)",
             color=INK, loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{outdir}/round2_events.png", dpi=150)
print("saved")
