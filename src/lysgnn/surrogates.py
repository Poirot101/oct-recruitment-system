"""Shared surrogate-model code for the round-2 studies (cross-validation, data scaling).

* `load_glass` reads a glass and its soft-mode terms.
* `softmode_table` builds the per-(site, direction) feature table used by gradient boosting.
* `train_gnn` / `predict_gnn` train and apply the GNN (raw or with soft-mode node features).
"""
import math

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from . import gnn


def load_glass(datadir, softdir, seed):
    d = dict(np.load(f"{datadir}/glass_{seed}.npz"))
    s = dict(np.load(f"{softdir}/softmode_{seed}.npz"))
    d["seed"] = seed
    d["lam"], d["kappa"], d["h"], d["soft_wall"] = s["lam"], s["kappa"], s["h"], float(s["wall"])
    d["y"] = (d["tauc"] - d["tau0"]).astype(np.float32)
    return d


def mode_yield(d):
    """Per-mode saddle-node strain gamma_k(theta), inf where the mode stiffens."""
    den = 2.0 * d["kappa"][:, None, :] * d["h"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den < 0, -d["lam"][:, None, :] ** 2 / den, np.inf)


def gmin(d):
    return np.minimum(mode_yield(d).min(2), 1.0)


def _neighbour_weights(pos, L, r):
    dd = pos[:, None, :] - pos[None, :, :]
    dd -= L * np.rint(dd / L)
    w = ((dd ** 2).sum(-1) < r * r).astype(float)
    return w / w.sum(1, keepdims=True)


def softmode_table(d):
    """Feature tensor [N, T, F] for gradient boosting (same features as hybrid_model.py)."""
    gk = mode_yield(d)
    lam, kap, h = d["lam"], d["kappa"], d["h"]
    order = np.argsort(gk, axis=2)
    take = lambda a: np.take_along_axis(a, order[:, :, :3], 2)
    g3 = np.minimum(take(gk), 1.0)
    lam3 = take(np.broadcast_to(lam[:, None, :], gk.shape))
    kap3 = take(np.broadcast_to(kap[:, None, :], gk.shape))
    h3 = take(np.abs(h))
    T = h.shape[1]
    tau0 = d["tau0"]
    Wn = _neighbour_weights(d["pos"], float(d["L"]), 1.5)
    loglam0 = np.log(np.maximum(lam[:, :1], 1e-6))
    cols = [np.log(g3[:, :, i]) for i in range(3)] + [lam3[:, :, i] for i in range(3)] + \
           [kap3[:, :, i] for i in range(3)] + [h3[:, :, i] for i in range(3)] + \
           [tau0, Wn @ np.log(g3[:, :, 0]), Wn @ tau0, np.repeat(Wn @ loglam0, T, 1),
            np.repeat(loglam0, T, 1), np.repeat(d["types"][:, None].astype(float), T, 1)]
    return np.stack(cols, -1)


def fit_gbr(train):
    tabs = [softmode_table(d) for d in train]
    X = np.concatenate([t.reshape(-1, t.shape[-1]) for t in tabs])
    y = np.concatenate([d["y"].ravel() for d in train])
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=0)
    return m.fit(X, y)


def predict_gbr(m, d):
    X = softmode_table(d)
    N, T, F = X.shape
    return m.predict(X.reshape(-1, F)).reshape(N, T)


def _sample(d, soft):
    x, ei, rel = gnn.build_graph(d["pos"], d["types"], float(d["L"]))
    xd = None
    if soft:
        xd = torch.tensor(np.stack([np.log(gmin(d)), d["tau0"] * 5.0], -1).astype(np.float32))
        x = np.concatenate([x, np.log(np.maximum(d["lam"][:, :3], 1e-6)).astype(np.float32)], 1)
    return dict(x=torch.tensor(x), xd=xd, ei=torch.tensor(ei), rel=torch.tensor(rel),
                types=torch.tensor(d["types"]), y=torch.tensor(d["y"]))


def _node_input(s, k=0):
    if s["xd"] is None:
        return s["x"]
    return torch.cat([s["x"], torch.roll(s["xd"], k, dims=1).reshape(s["x"].shape[0], -1)], 1)


def train_gnn(train, val, soft, epochs=300, layers=4, hidden=64, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = [_sample(d, soft) for d in train]
    va = [_sample(d, soft) for d in val]
    T = tr[0]["y"].shape[1]
    scale = float(torch.cat([s["y"] for s in tr]).std())
    model = gnn.LYSNet(n_out=T, hidden=hidden, layers=layers, n_in=_node_input(tr[0]).shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def loss(s, k=0):
        rel = gnn.rotate(s["rel"], k, T) if k else s["rel"]
        e = gnn.edge_features(rel, s["types"], s["ei"])
        p = model(_node_input(s, k), s["ei"], e) * scale
        return torch.mean((p - torch.roll(s["y"], k, dims=1)) ** 2)

    best, state = math.inf, None
    for _ in range(epochs):
        model.train()
        for i in rng.permutation(len(tr)):
            opt.zero_grad()
            loss(tr[i], int(rng.integers(T))).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss(s).item() for s in va]))
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return model, scale, soft


def predict_gnn(trained, d):
    model, scale, soft = trained
    s = _sample(d, soft)
    with torch.no_grad():
        e = gnn.edge_features(s["rel"], s["types"], s["ei"])
        return model(_node_input(s), s["ei"], e).numpy() * scale
