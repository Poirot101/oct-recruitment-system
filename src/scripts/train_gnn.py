"""Train the GNN surrogate on local-yield-stress maps and evaluate it.

Usage: python scripts/train_gnn.py DATADIR OUT.json --train 1 2 3 --val 4 --test 5 6
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from lysgnn import gnn

ap = argparse.ArgumentParser()
ap.add_argument("datadir")
ap.add_argument("out")
ap.add_argument("--train", type=int, nargs="+")
ap.add_argument("--val", type=int, nargs="+")
ap.add_argument("--test", type=int, nargs="+")
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--first-events", type=int, default=10)
args = ap.parse_args()
torch.manual_seed(0)
torch.set_num_threads(4)


def load(seed):
    d = dict(np.load(f"{args.datadir}/glass_{seed}.npz"))
    x, ei, rel = gnn.build_graph(d["pos"], d["types"], float(d["L"]))
    return dict(seed=seed, raw=d, x=torch.tensor(x), ei=torch.tensor(ei), rel=torch.tensor(rel),
                types=torch.tensor(d["types"]),
                y=torch.tensor((d["tauc"] - d["tau0"]).astype(np.float32)))


train = [load(s) for s in args.train]
val = [load(s) for s in args.val]
test = [load(s) for s in args.test]
n_theta = train[0]["y"].shape[1]
scale = float(torch.cat([s["y"] for s in train]).std())

model = gnn.LYSNet(n_out=n_theta, hidden=args.hidden, layers=args.layers)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)


def loss_on(s, k=0):
    rel = gnn.rotate(s["rel"], k, n_theta) if k else s["rel"]
    e = gnn.edge_features(rel, s["types"], s["ei"])
    pred = model(s["x"], s["ei"], e) * scale
    y = torch.roll(s["y"], k, dims=1)
    return torch.mean((pred - y) ** 2)


best, best_state, hist = 1e9, None, []
t0 = time.perf_counter()
for ep in range(args.epochs):
    model.train()
    for i in np.random.permutation(len(train)):
        opt.zero_grad()
        loss = loss_on(train[i], int(np.random.randint(n_theta)))
        loss.backward()
        opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        vl = float(np.mean([loss_on(s).item() for s in val]))
    hist.append(vl)
    if vl < best:
        best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    if ep % 25 == 0:
        print(f"epoch {ep} val_mse {vl:.5f}", flush=True)
train_wall = time.perf_counter() - t0
model.load_state_dict(best_state)
model.eval()
torch.save(best_state, args.out.replace(".json", ".pt"))


def r2(a, b):
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())


def event_scores(field_soft, ev_atom, n_first):
    """Percentile rank (0 = softest) of each early event's atom in a softness field."""
    order = np.argsort(np.argsort(field_soft))  # 0 = lowest value = softest
    ranks = order[ev_atom[:n_first]] / (len(field_soft) - 1)
    return ranks


def smooth(field, pos, L, r=2.5):
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.rint(d / L)
    w = ((d ** 2).sum(-1) < r * r).astype(float)
    return (w @ field) / w.sum(1)


report = {"train": args.train, "val": args.val, "test": args.test, "epochs": args.epochs,
          "train_wall_s": train_wall, "best_val_mse": best, "glasses": []}
for s in test:
    t1 = time.perf_counter()
    pred = gnn.predict(model, s) * scale
    infer = time.perf_counter() - t1
    y = s["y"].numpy()
    raw = s["raw"]
    g = {"seed": s["seed"], "gnn_inference_s": infer, "lys_wall_s": float(raw["lys_wall"]),
         "lys_minimiser_iterations": int(raw["lys_iterations"].sum()),
         "r2_all": r2(y.ravel(), pred.ravel()),
         "spearman_all": float(spearmanr(y.ravel(), pred.ravel())[0]),
         "r2_min_theta": r2(y.min(1), pred.min(1)),
         "spearman_min_theta": float(spearmanr(y.min(1), pred.min(1))[0]),
         "spearman_theta0": float(spearmanr(y[:, 0], pred[:, 0])[0])}
    if "ev_atom" in raw and len(raw["ev_atom"]):
        pos, L = raw["pos"], float(raw["L"])
        fields = {"lys_true": y[:, 0], "gnn": pred[:, 0],
                  "epot_smoothed": -smooth(raw["epot"], pos, L),
                  "random": np.random.default_rng(s["seed"]).random(len(pos))}
        ev = raw["ev_atom"]
        g["n_events"] = int(len(ev))
        g["events"] = {}
        for name, f in fields.items():
            rk = event_scores(f, ev, args.first_events)
            rk_all = event_scores(f, ev, len(ev))
            g["events"][name] = {"mean_pct_first": float(rk.mean()),
                                 "frac_in_softest10_first": float((rk < 0.1).mean()),
                                 "mean_pct_all": float(rk_all.mean()),
                                 "frac_in_softest10_all": float((rk_all < 0.1).mean())}
    report["glasses"].append(g)
    np.savez_compressed(args.out.replace(".json", f"_pred_{s['seed']}.npz"), pred=pred, y=y)
    print(json.dumps(g), flush=True)
json.dump(report, open(args.out, "w"), indent=1)
