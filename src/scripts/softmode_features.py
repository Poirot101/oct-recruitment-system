"""Compute soft-mode terms (lambda_k, kappa_k, h_k(theta)) for every atom of each glass.

Usage: python scripts/softmode_features.py DATADIR OUTDIR SEED [SEED ...]
"""
import sys
import time

import numpy as np

from lysgnn import glass, sim

datadir, outdir, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
N_MODES = 12
for seed in seeds:
    d = np.load(f"{datadir}/glass_{seed}.npz")
    pos, types, L = d["pos"], d["types"], float(d["L"])
    centers = np.arange(len(pos))
    sim.softmode_terms_all(pos, types, L, centers[:2], d["thetas"], 5.0, N_MODES, *glass.P)
    t0 = time.perf_counter()
    W, K, Hh = sim.softmode_terms_all(pos, types, L, centers, d["thetas"], 5.0, N_MODES, *glass.P)
    wall = time.perf_counter() - t0
    np.savez_compressed(f"{outdir}/softmode_{seed}.npz", lam=W, kappa=K, h=Hh, wall=wall)
    print(f"seed {seed}: {wall:.1f}s", flush=True)
