"""Extra adaptive-stepping settings against the saved baseline (reuses benchmark_lys output).

Usage: python scripts/benchmark_lys_more.py GLASS.npz BASELINE.npz OUT.json
"""
import json
import sys

import numpy as np

from lysgnn import glass

d = np.load(sys.argv[1])
b = np.load(sys.argv[2])
pos, types, L = d["pos"], d["types"], float(d["L"])
centers = b["centers"]
base_wall = json.load(open(sys.argv[2].replace("_baseline.npz", ".json")))["baseline"]["wall_s"]
glass.lys_map(pos, types, L, centers=centers[:1], mode=1)
res = []
for dgc in [4e-3, 8e-3, 1.6e-2, 3.2e-2]:
    for sf in [0.8, 0.9]:
        acc = glass.lys_map(pos, types, L, centers=centers, mode=1, dg_coarse=dgc, slope_frac=sf)
        err = np.abs(acc["tauc"] - b["tauc"])
        res.append({"dg_coarse": dgc, "slope_frac": sf, "wall_s": acc["wall"],
                    "speedup_wall": base_wall / acc["wall"],
                    "speedup_iterations": float(b["iterations"].sum() / acc["iterations"].sum()),
                    "frac_tauc_within_1e-3": float((err < 1e-3).mean()),
                    "frac_min_over_theta_within_1e-3": float(
                        (np.abs(acc["tauc"].min(1) - b["tauc"].min(1)) < 1e-3).mean())})
        print(res[-1], flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
