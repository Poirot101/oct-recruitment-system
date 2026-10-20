"""Figures for analysis/ANALYSIS.md (run from repo root after the experiments)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e2dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})

bench = json.load(open("results/benchmark_lys.json"))
base_site = bench["baseline"]["wall_s"] / bench["n_sites"]
exact = [a for a in bench["accelerated"] if a["dg_coarse"] == 1e-3 and a["slope_frac"] == 0.8][0]
lys_wall = np.mean([float(np.load(f"results/data/glass_{s}.npz")["lys_wall"]) for s in range(10, 18)])
soft_wall = np.mean([float(np.load(f"results/softmode/softmode_{s}.npz")["wall"]) for s in range(10, 18)])
gh = json.load(open("results/gnn_hybrid.json"))
g = json.load(open("results/gnn.json"))
hb = json.load(open("results/hybrid.json"))
mean = lambda rep, k, sub=None: float(np.mean([(x[sub] if sub else x)[k] for x in rep["glasses"]]))

rows = [  # label, seconds per 1024-site map (6 directions, 4 cores), Spearman vs exact map
    ("Baseline LYS scan", base_site * 1024, 1.0),
    ("Adaptive LYS (exact setting)", exact["wall_s"] / bench["n_sites"] * 1024, 1.0),
    ("Adaptive LYS (fast setting)", lys_wall, 0.997),
    ("Soft-mode estimate, no learning", soft_wall, mean(hb, "spearman_all", "softmode_raw")),
    ("Soft modes + gradient boosting", soft_wall, mean(hb, "spearman_all", "hybrid")),
    ("Soft modes + GNN", soft_wall + 0.1, mean(gh, "spearman_all")),
    ("GNN on raw structure", 0.1, mean(g, "spearman_all")),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
y = np.arange(len(rows))[::-1]
axes[0].barh(y, [r[1] for r in rows], color=BLUE, height=0.55)
axes[0].set_xscale("log")
axes[0].set_xlabel("wall time per glass map, s (log scale)")
axes[0].set_yticks(y, [r[0] for r in rows])
for yi, r in zip(y, rows):
    axes[0].text(r[1] * 1.15, yi, f"{r[1]:.3g} s", va="center", color=INK, fontsize=9)
axes[0].set_title("Cost", color=INK, loc="left")
axes[1].barh(y, [max(r[2], 0) for r in rows], color=BLUE, height=0.55)
axes[1].set_xlim(0, 1.15)
axes[1].set_xlabel("Spearman rank correlation with exact map")
for yi, r in zip(y, rows):
    axes[1].text(max(r[2], 0) + 0.02, yi, f"{r[2]:.2f}", va="center", color=INK, fontsize=9)
axes[1].set_title("Accuracy (all sites x directions)", color=INK, loc="left")
for ax in axes:
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("analysis/figures/cost_vs_accuracy.png", dpi=160)

# maps: exact vs hybrid-GNN predicted min-over-direction yield stress increment, seed 16
d = np.load("results/data/glass_16.npz")
p = np.load("results/gnn_hybrid_pred_16.npz")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
vals = [p["y"].min(1), p["pred"].min(1)]
lo, hi = np.quantile(vals[0], [0.02, 0.98])
for ax, v, t in zip(axes, vals, ["Exact local yield stress scan", "Soft modes + GNN surrogate"]):
    sc = ax.scatter(d["pos"][:, 0], d["pos"][:, 1], c=v, s=9, cmap="Blues", vmin=lo, vmax=hi)
    ev = d["ev_atom"][:10]
    ax.scatter(d["pos"][ev, 0], d["pos"][ev, 1], s=60, facecolors="none", edgecolors="#eb6834",
               linewidths=1.5, label="first 10 plastic events (global AQS)")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(t, color=INK, loc="left")
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.02), frameon=False, fontsize=9)
cb = fig.colorbar(sc, ax=axes, shrink=0.8)
cb.set_label("min over directions of Δτ_c (light = weak)")
fig.savefig("analysis/figures/lys_maps_seed16.png", dpi=160, bbox_inches="tight")
print("ok")

# screening: how many of the truly weakest 10% of sites (min over directions)
# fall inside the predicted weakest fraction f of sites
screen = {}
for name, pat in [("softmodes_gnn", "results/gnn_hybrid_pred_%d.npz"),
                  ("softmodes_gbr", "results/hybrid_pred_%d.npz"),
                  ("gnn_raw", "results/gnn_pred_%d.npz")]:
    screen[name] = {}
    for s in (16, 17):
        p = np.load(pat % s)
        yt, q = p["y"].min(1), p["pred"].min(1)
        n = len(yt)
        weak = set(np.argsort(yt)[: n // 10])
        screen[name][s] = {f: len(weak & set(np.argsort(q)[: int(f * n)])) / len(weak)
                           for f in (0.1, 0.2, 0.3, 0.5)}
json.dump(screen, open("results/screening.json", "w"), indent=1)
