"""Round-2 study: data scaling, cross-validated surrogates, plastic-event test, screening.

Usage: python scripts/round2_study.py DATADIR SOFTDIR OUT.json
"""
import json
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from lysgnn import surrogates as S

datadir, softdir, out = sys.argv[1:4]
SEEDS = list(range(10, 30))
TEST = [16, 17, 26, 27, 28, 29]
VAL = [15]
POOL = [s for s in SEEDS if s not in TEST + VAL]
G = {s: S.load_glass(datadir, softdir, s) for s in SEEDS}
rng = np.random.default_rng(0)


def scores(d, pred):
    y = d["y"]
    n = len(y)
    weak = set(np.argsort(y.min(1))[: n // 10])
    order = np.argsort(pred.min(1))
    rec = {f: len(weak & set(order[: int(f * n)])) / len(weak) for f in (0.1, 0.2, 0.3, 0.5)}
    return {"spearman_all": float(spearmanr(y.ravel(), pred.ravel())[0]),
            "spearman_min_theta": float(spearmanr(y.min(1), pred.min(1))[0]),
            "recall_weakest10": rec}


def mean_scores(lst):
    return {"spearman_all": float(np.mean([x["spearman_all"] for x in lst])),
            "spearman_min_theta": float(np.mean([x["spearman_min_theta"] for x in lst])),
            "recall_weakest10": {f: float(np.mean([x["recall_weakest10"][f] for x in lst]))
                                 for f in (0.1, 0.2, 0.3, 0.5)}}


report = {"seeds": SEEDS, "test": TEST, "val": VAL, "pool": POOL}

# 1. data scaling on a fixed test set ---------------------------------------
scaling = []
for n in (2, 4, 8, len(POOL)):
    train = [G[s] for s in POOL[:n]]
    row = {"n_train_glasses": n}
    t0 = time.perf_counter()
    m = S.fit_gbr(train)
    row["gbr"] = mean_scores([scores(G[s], S.predict_gbr(m, G[s])) for s in TEST])
    row["gbr_fit_s"] = time.perf_counter() - t0
    for name, soft in (("gnn_raw", False), ("gnn_soft", True)):
        t0 = time.perf_counter()
        tr = S.train_gnn(train, [G[s] for s in VAL], soft)
        row[name] = mean_scores([scores(G[s], S.predict_gnn(tr, G[s])) for s in TEST])
        row[name + "_fit_s"] = time.perf_counter() - t0
    row["softmode_raw"] = mean_scores([scores(G[s], S.gmin(G[s])) for s in TEST])
    scaling.append(row)
    print(json.dumps(row), flush=True)
report["scaling"] = scaling

# 2. 5-fold cross-validated predictions for every glass -----------------------
folds = np.array_split(rng.permutation(SEEDS), 5)
cv = {}
for k, fold in enumerate(folds):
    rest = [s for s in SEEDS if s not in fold]
    val, train = rest[:1], rest[1:]
    m = S.fit_gbr([G[s] for s in train])
    tr = S.train_gnn([G[s] for s in train], [G[s] for s in val], True, seed=k)
    for s in fold:
        s = int(s)
        cv[s] = {"gbr": S.predict_gbr(m, G[s]), "gnn_soft": S.predict_gnn(tr, G[s])}
    print(f"fold {k} done", flush=True)
for s in SEEDS:
    np.savez_compressed(out.replace(".json", f"_cv_{s}.npz"), **cv[s])
report["cv"] = {name: mean_scores([scores(G[s], cv[s][name]) for s in SEEDS])
                for name in ("gbr", "gnn_soft")}
report["cv"]["softmode_raw"] = mean_scores([scores(G[s], S.gmin(G[s])) for s in SEEDS])
report["cv"]["epot"] = mean_scores([scores(G[s], np.repeat(-G[s]["epot"][:, None], 6, 1))
                                    for s in SEEDS])

# 3. plastic events (first 10 per glass, theta = 0 field, percentile 0 = weakest) ---
FIRST = 10
fields = {"lys_exact": lambda s: G[s]["y"][:, 0], "gnn_soft": lambda s: cv[s]["gnn_soft"][:, 0],
          "gbr": lambda s: cv[s]["gbr"][:, 0], "softmode_raw": lambda s: S.gmin(G[s])[:, 0],
          "epot": lambda s: -G[s]["epot"],
          "random": lambda s: np.random.default_rng(s).random(len(G[s]["y"]))}
ev_glasses = [s for s in SEEDS if "ev_atom" in G[s] and len(G[s]["ev_atom"]) >= FIRST]
pct = {name: [] for name in fields}
for s in ev_glasses:
    ev = G[s]["ev_atom"][:FIRST]
    for name, f in fields.items():
        v = f(s)
        r = np.argsort(np.argsort(v)) / (len(v) - 1)
        pct[name].append(r[ev])
events = {"n_glasses": len(ev_glasses), "n_events": len(ev_glasses) * FIRST}
for name, per_glass in pct.items():
    per_glass = np.array(per_glass)  # [glasses, FIRST]
    boot = [per_glass[rng.integers(0, len(per_glass), len(per_glass))].mean()
            for _ in range(2000)]
    events[name] = {"mean_percentile": float(per_glass.mean()),
                    "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                    "frac_in_weakest10": float((per_glass < 0.1).mean()),
                    "frac_in_weakest30": float((per_glass < 0.3).mean())}
report["events"] = events
print(json.dumps(events, indent=1), flush=True)

# 4. screen-then-verify cost model ----------------------------------------------
lys_wall = float(np.mean([G[s]["lys_wall"] for s in SEEDS if s >= 18]))
soft_wall = float(np.mean([G[s]["soft_wall"] for s in SEEDS]))
report["screening"] = {"lys_wall_s": lys_wall, "softmode_wall_s": soft_wall,
                       "options": [{"verify_fraction": f,
                                    "recall_weakest10": report["cv"]["gnn_soft"]["recall_weakest10"][f],
                                    "wall_s": soft_wall + f * lys_wall}
                                   for f in (0.1, 0.2, 0.3, 0.5)]}
json.dump(report, open(out, "w"), indent=1)
print("saved", out)
