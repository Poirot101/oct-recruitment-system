"""Benchmark the accelerated local-yield-stress scan against the baseline.

Usage: python scripts/benchmark_lys.py GLASS.npz N_SITES OUT.json
Runs the baseline (fixed dgamma = 1e-4) and the accelerated scan with several
coarse-step settings on the same random sites and reports wall time, minimiser
iterations, and agreement of tau_c with the baseline.
"""
import json
import sys

import numpy as np

from lysgnn import glass

path, n_sites, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = np.load(path)
pos, types, L = d["pos"], d["types"], float(d["L"])
rng = np.random.default_rng(0)
centers = rng.choice(len(pos), n_sites, replace=False)
glass.lys_map(pos, types, L, centers=centers[:1], mode=0)  # JIT warm-up
glass.lys_map(pos, types, L, centers=centers[:1], mode=1)

base = glass.lys_map(pos, types, L, centers=centers, mode=0)
res = {"n_sites": n_sites, "n_theta": int(base["tauc"].shape[1]),
       "baseline": {"wall_s": base["wall"], "iterations": int(base["iterations"].sum())},
       "accelerated": []}
for dgc in [1e-3, 2e-3, 4e-3]:
    for sf in [0.5, 0.8]:
        acc = glass.lys_map(pos, types, L, centers=centers, mode=1, dg_coarse=dgc, slope_frac=sf)
        err = np.abs(acc["tauc"] - base["tauc"])
        res["accelerated"].append({
            "dg_coarse": dgc, "slope_frac": sf, "wall_s": acc["wall"],
            "iterations": int(acc["iterations"].sum()),
            "speedup_wall": base["wall"] / acc["wall"],
            "speedup_iterations": base["iterations"].sum() / acc["iterations"].sum(),
            "frac_tauc_within_1e-3": float((err < 1e-3).mean()),
            "frac_min_over_theta_within_1e-3": float(
                (np.abs(acc["tauc"].min(1) - base["tauc"].min(1)) < 1e-3).mean()),
            "max_abs_err": float(err.max()),
        })
        print(res["accelerated"][-1], flush=True)
np.savez(out.replace(".json", "_baseline.npz"), centers=centers, **{k: base[k] for k in ["tau0", "tauc", "gammac", "iterations"]})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["baseline"]))
