"""Add global AQS plastic events to existing glass files that lack them.

Usage: python scripts/add_aqs.py DATADIR SEED [SEED ...]
"""
import sys
import time

import numpy as np

from lysgnn import glass

datadir, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
for seed in seeds:
    path = f"{datadir}/glass_{seed}.npz"
    d = dict(np.load(path))
    if "ev_atom" in d:
        continue
    t1 = time.perf_counter()
    a = glass.global_aqs(d["pos"], d["types"], float(d["L"]), dgamma=1e-4, gmax=0.1)
    d.update(aqs_gamma=a["gamma"], aqs_stress=a["stress"], aqs_iterations=a["iterations"],
             aqs_wall=time.perf_counter() - t1,
             ev_gamma=np.array([e["gamma"] for e in a["events"]]),
             ev_atom=np.array([e["atom"] for e in a["events"]], np.int64),
             ev_drop=np.array([e["drop"] for e in a["events"]]))
    np.savez_compressed(path, **d)
    print(f"seed {seed}: {len(a['events'])} events", flush=True)
