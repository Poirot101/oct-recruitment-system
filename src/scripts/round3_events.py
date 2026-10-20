"""Round-3 event study: is per-atom energy really as predictive as the exact LYS map?

Re-runs the global AQS shear of every glass up to its first 10 plastic events, keeping the
full non-affine displacement field of each event, and then asks:
  * how strongly the "largest non-affine displacement" locator is biased by particle type,
  * whether energy still predicts events within each particle type,
  * how the predictors rank when the event is located by its neighbourhood-averaged centre,
  * whether combining energy with the LYS / soft-mode fields beats either alone.

Usage: python scripts/round3_events.py DATADIR SOFTDIR CVPREFIX OUT.json [--workers 4]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

SEEDS = list(range(10, 30))
FIRST = 10
datadir, softdir, cvprefix, out = sys.argv[1:5]
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
cache = os.path.join(os.path.dirname(out), "events")


def rerun(seed):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from lysgnn import glass
    path = f"{cache}/events_{seed}.npz"
    if os.path.exists(path):
        return path
    d = np.load(f"{datadir}/glass_{seed}.npz")
    gmax = float(d["ev_gamma"][FIRST - 1]) + 5e-4
    a = glass.global_aqs(d["pos"], d["types"], float(d["L"]), dgamma=1e-4, gmax=gmax)
    ev = a["events"][:FIRST]
    np.savez_compressed(path, atom=np.array([e["atom"] for e in ev]),
                        gamma=np.array([e["gamma"] for e in ev]),
                        nonaffine=np.stack([e["nonaffine"] for e in ev]))
    return path


def neighbour_weights(pos, L, r):
    dd = pos[:, None, :] - pos[None, :, :]
    dd -= L * np.rint(dd / L)
    w = ((dd ** 2).sum(-1) < r * r).astype(float)
    return w / w.sum(1, keepdims=True), w > 0


def pct(v):
    """Percentile rank in [0, 1]; 0 = predicted weakest."""
    return (rankdata(v) - 1) / (len(v) - 1)


def within_type_z(v, types):
    z = np.empty_like(v, dtype=float)
    for t in (0, 1):
        m = types == t
        z[m] = (v[m] - v[m].mean()) / v[m].std()
    return z


if __name__ == "__main__":
    from lysgnn import surrogates as S

    os.makedirs(cache, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(rerun, SEEDS))

    rows = {}
    stats = {"type0_fraction_atoms": [], "type0_fraction_events_atom": [],
             "type0_fraction_events_centre": [], "same_locator_fraction": [],
             "epot_mean_by_type": [], "centre_to_atom_distance": []}
    for s in SEEDS:
        G = S.load_glass(datadir, softdir, s)
        cv = np.load(f"{cvprefix}_cv_{s}.npz")
        e = np.load(f"{cache}/events_{s}.npz")
        types, pos, L = G["types"], G["pos"], float(G["L"])
        W, _ = neighbour_weights(pos, L, 1.5)
        _, near2 = neighbour_weights(pos, L, 2.0)
        atom = e["atom"]
        centre = np.array([int(np.argmax(W @ na)) for na in e["nonaffine"]])
        dd = pos[centre] - pos[atom]
        dd -= L * np.rint(dd / L)
        stats["centre_to_atom_distance"].append(float(np.sqrt((dd ** 2).sum(1)).mean()))
        stats["type0_fraction_atoms"].append(float((types == 0).mean()))
        stats["type0_fraction_events_atom"].append(float((types[atom] == 0).mean()))
        stats["type0_fraction_events_centre"].append(float((types[centre] == 0).mean()))
        stats["same_locator_fraction"].append(float((atom == centre).mean()))
        stats["epot_mean_by_type"].append([float(G["epot"][types == t].mean()) for t in (0, 1)])

        y = G["y"]
        ez = -within_type_z(G["epot"], types)
        fields = {
            "lys_theta0": y[:, 0], "lys_min_theta": y.min(1),
            "lys_theta0_smoothed": W @ y[:, 0],
            "epot": -G["epot"], "epot_within_type": ez,
            "type_only": types.astype(float) + 1e-6 * np.random.default_rng(s).random(len(y)),
            "softmode_raw": S.gmin(G)[:, 0], "gbr": cv["gbr"][:, 0], "gnn_soft": cv["gnn_soft"][:, 0],
            "lys_theta0+epot": pct(y[:, 0]) + pct(ez),
            "softmode+epot": pct(S.gmin(G)[:, 0]) + pct(ez),
            "gbr+epot": pct(cv["gbr"][:, 0]) + pct(ez),
            "random": np.random.default_rng(s).random(len(y)),
        }
        for name, v in fields.items():
            p = pct(v)
            r = rows.setdefault(name, {"atom": [], "centre": [], "within_type_atom": [],
                                       "nbhd_min_centre": []})
            r["atom"].append(p[atom])
            r["centre"].append(p[centre])
            wt = np.empty(len(v))
            for t in (0, 1):
                m = types == t
                wt[m] = pct(v[m])
            r["within_type_atom"].append(wt[atom])
            # best (weakest) predicted site within r = 2 of the event centre, as a percentile
            r["nbhd_min_centre"].append(np.array([p[near2[c]].min() for c in centre]))

    rng = np.random.default_rng(0)
    report = {"n_glasses": len(SEEDS), "n_events": len(SEEDS) * FIRST,
              "locator": {k: float(np.mean(v, axis=0).mean()) if k != "epot_mean_by_type"
                          else np.mean(v, axis=0).tolist() for k, v in stats.items()}}
    for name, r in rows.items():
        report[name] = {}
        for key, per_glass in r.items():
            a = np.array(per_glass)
            boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
            report[name][key] = {"mean_percentile": float(a.mean()),
                                 "ci95": [float(np.quantile(boot, 0.025)),
                                          float(np.quantile(boot, 0.975))],
                                 "frac_in_weakest10": float((a < 0.1).mean())}
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps(report["locator"], indent=1))
    for name in rows:
        print(f"{name:22s}", "  ".join(f"{k}={report[name][k]['mean_percentile']:.3f}"
                                       for k in rows[name]))
