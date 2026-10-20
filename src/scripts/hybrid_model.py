"""Hybrid physics + ML surrogate: soft-mode features -> gradient boosting -> Delta tau_c.

Usage: python scripts/hybrid_model.py DATADIR SOFTDIR OUT.json --train ... --test ...

Each (site, direction) is one sample, so the model is shared across loading
directions. Features come from the frozen-boundary cluster's Hessian (soft
modes) plus the local pre-stress tau_0, and their averages over neighbouring
sites. Also reports the raw soft-mode estimate (no learning) and the GNN-free
baselines, and scores every predictor against the plastic events of global AQS.
"""
import argparse
import json
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ap = argparse.ArgumentParser()
ap.add_argument("datadir")
ap.add_argument("softdir")
ap.add_argument("out")
ap.add_argument("--train", type=int, nargs="+")
ap.add_argument("--test", type=int, nargs="+")
ap.add_argument("--first-events", type=int, default=10)
args = ap.parse_args()


def neighbours(pos, L, r):
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.rint(d / L)
    return ((d ** 2).sum(-1) < r * r).astype(float)


def features(seed):
    d = dict(np.load(f"{args.datadir}/glass_{seed}.npz"))
    s = np.load(f"{args.softdir}/softmode_{seed}.npz")
    lam, kap, h = s["lam"], s["kappa"], s["h"]  # [N,m], [N,m], [N,T,m]
    N, T, m = h.shape
    den = 2.0 * kap[:, None, :] * h
    with np.errstate(divide="ignore", invalid="ignore"):
        gk = np.where(den < 0, -lam[:, None, :] ** 2 / den, np.inf)
    order = np.argsort(gk, axis=2)
    take = lambda a: np.take_along_axis(a, order[:, :, :3], 2)
    g3 = np.minimum(take(gk), 1.0)
    lam3 = take(np.broadcast_to(lam[:, None, :], gk.shape))
    kap3 = take(np.broadcast_to(kap[:, None, :], gk.shape))
    h3 = take(np.abs(h))
    gmin = g3[:, :, 0]
    tau0 = d["tau0"]
    W = neighbours(d["pos"], float(d["L"]), 1.5)
    Wn = W / W.sum(1, keepdims=True)
    nb_gmin = Wn @ np.log(gmin)
    nb_lam0 = Wn @ np.log(np.maximum(lam[:, :1], 1e-6))
    nb_tau0 = Wn @ tau0
    cols = [np.log(g3[:, :, i]) for i in range(3)] + [lam3[:, :, i] for i in range(3)] + \
           [kap3[:, :, i] for i in range(3)] + [h3[:, :, i] for i in range(3)] + \
           [tau0, nb_gmin, nb_tau0, np.repeat(nb_lam0, T, 1),
            np.repeat(np.log(np.maximum(lam[:, :1], 1e-6)), T, 1),
            np.repeat(d["types"][:, None].astype(float), T, 1)]
    X = np.stack(cols, -1)  # [N, T, F]
    y = d["tauc"] - d["tau0"]
    return d, X, y, gmin, float(s["wall"])


t0 = time.perf_counter()
train = [features(s) for s in args.train]
Xtr = np.concatenate([t[1].reshape(-1, t[1].shape[-1]) for t in train])
ytr = np.concatenate([t[2].ravel() for t in train])
model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=0)
model.fit(Xtr, ytr)
fit_wall = time.perf_counter() - t0


def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())


def ranks_of_events(field_soft, ev, n):
    order = np.argsort(np.argsort(field_soft))
    return order[ev[:n]] / (len(field_soft) - 1)


report = {"train": args.train, "test": args.test, "fit_wall_s": fit_wall, "n_train_rows": int(len(ytr)),
          "glasses": []}
for seed in args.test:
    d, X, y, gmin, soft_wall = features(seed)
    N, T, F = X.shape
    t1 = time.perf_counter()
    pred = model.predict(X.reshape(-1, F)).reshape(N, T)
    infer = time.perf_counter() - t1
    g = {"seed": seed, "lys_wall_s": float(d["lys_wall"]), "softmode_wall_s": soft_wall,
         "model_inference_s": infer,
         "hybrid": {"r2_all": r2(y.ravel(), pred.ravel()),
                    "spearman_all": float(spearmanr(y.ravel(), pred.ravel())[0]),
                    "spearman_min_theta": float(spearmanr(y.min(1), pred.min(1))[0]),
                    "r2_min_theta": r2(y.min(1), pred.min(1)),
                    "spearman_theta0": float(spearmanr(y[:, 0], pred[:, 0])[0])},
         "softmode_raw": {"spearman_all": float(spearmanr(y.ravel(), gmin.ravel())[0]),
                          "spearman_min_theta": float(spearmanr(y.min(1), gmin.min(1))[0]),
                          "spearman_theta0": float(spearmanr(y[:, 0], gmin[:, 0])[0])}}
    if "ev_atom" in d and len(d["ev_atom"]):
        ev = d["ev_atom"]
        rng = np.random.default_rng(seed)
        fields = {"lys_true": y[:, 0], "hybrid": pred[:, 0], "softmode_raw": gmin[:, 0],
                  "epot": -d["epot"], "random": rng.random(N)}
        g["n_events"] = int(len(ev))
        g["events"] = {}
        for name, f in fields.items():
            rf = ranks_of_events(f, ev, args.first_events)
            ra = ranks_of_events(f, ev, len(ev))
            g["events"][name] = {"mean_pct_first": float(rf.mean()),
                                 "frac_softest10_first": float((rf < 0.1).mean()),
                                 "mean_pct_all": float(ra.mean()),
                                 "frac_softest10_all": float((ra < 0.1).mean())}
    np.savez_compressed(args.out.replace(".json", f"_pred_{seed}.npz"), pred=pred, y=y, gmin=gmin)
    report["glasses"].append(g)
    print(json.dumps(g, indent=1), flush=True)
json.dump(report, open(args.out, "w"), indent=1)
