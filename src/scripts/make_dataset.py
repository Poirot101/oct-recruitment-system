"""Generate glasses, their full local-yield-stress maps, and global AQS events.

Usage: python scripts/make_dataset.py OUTDIR SEED [SEED ...] [--aqs]
Each seed gives OUTDIR/glass_<seed>.npz. With --aqs, the glass is also sheared
globally (AQS, dgamma = 1e-4 up to gamma = 0.1) and its plastic events are stored.
"""
import sys
import time

import numpy as np

from lysgnn import glass, sim

args = [a for a in sys.argv[1:] if not a.startswith("--")]
do_aqs = "--aqs" in sys.argv
outdir, seeds = args[0], [int(s) for s in args[1:]]

for seed in seeds:
    t0 = time.perf_counter()
    pos, types, L, info = glass.make_glass(n=1024, density=1.0, n_equil=10000,
                                           n_quench=20000, seed=seed)
    t_gen = time.perf_counter() - t0
    lys = glass.lys_map(pos, types, L, mode=1, dg_coarse=4e-3, slope_frac=0.8)
    pi, pj = sim.build_nlist(pos, L, 0.0, sim.RCMAX + 0.4)
    epot = sim.per_atom_energy(pos, types, pi, pj, L, 0.0, *glass.P)
    out = dict(pos=pos, types=types, L=L, tau0=lys["tau0"], tauc=lys["tauc"],
               gammac=lys["gammac"], lys_iterations=lys["iterations"], lys_wall=lys["wall"],
               thetas=lys["thetas"], epot=epot, gen_wall=t_gen)
    if do_aqs:
        t1 = time.perf_counter()
        a = glass.global_aqs(pos, types, L, dgamma=1e-4, gmax=0.1)
        out.update(aqs_gamma=a["gamma"], aqs_stress=a["stress"], aqs_iterations=a["iterations"],
                   aqs_wall=time.perf_counter() - t1,
                   ev_gamma=np.array([e["gamma"] for e in a["events"]]),
                   ev_atom=np.array([e["atom"] for e in a["events"]], np.int64),
                   ev_drop=np.array([e["drop"] for e in a["events"]]))
    np.savez_compressed(f"{outdir}/glass_{seed}.npz", **out)
    print(f"seed {seed}: gen {t_gen:.0f}s, lys {lys['wall']:.0f}s, "
          f"P={info['pressure']:.3f}, e={info['energy']:.4f}, "
          f"events={len(out.get('ev_atom', []))}", flush=True)
