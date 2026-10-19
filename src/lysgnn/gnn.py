"""Edge-conditioned graph neural network surrogate for local yield stress maps.

The design follows the lessons of GFF-MAP (Loiodice et al., npj Comput. Mater.
12:275, 2026): the material is a graph (here atoms are nodes and neighbouring
atoms are joined by edges), the relative position vector is the key edge
feature, and the convolution weights depend on the edge attributes (GFF-MAP
uses GMMConv, Monti et al. 2017; here an NNConv-style gate, Gilmer et al. 2017,
which needs no extra library). A node-wise linear head predicts the local
yield stress increment Delta tau_c(theta_k) = tau_c - tau_0 for the n_theta
loading directions of the scan.

Rotating a configuration by pi/n_theta maps direction k onto k+1, so random
rotations by multiples of pi/n_theta with the matching cyclic shift of the
targets are an exact data augmentation.

Pure PyTorch (index_add_ for the message aggregation), CPU friendly.
"""

import math

import numpy as np
import torch
from torch import nn

from . import sim

EDGE_CUT = 2.0


def build_graph(pos, types, L, cut=EDGE_CUT):
    """Return (node_feat [N,2], edge_index [2,E], rel [E,2]) for a periodic glass."""
    pi, pj = sim.build_nlist(pos, L, 0.0, cut)
    d = pos[pi] - pos[pj]
    d -= L * np.rint(d / L)
    src = np.concatenate([pj, pi])  # message j -> i
    dst = np.concatenate([pi, pj])
    rel = np.concatenate([d, -d])  # r_i - r_j as seen from receiver i
    x = np.eye(2, dtype=np.float32)[types]
    return x, np.stack([src, dst]).astype(np.int64), rel.astype(np.float32)


def edge_features(rel, types, edge_index, n_rbf=16, cut=EDGE_CUT):
    """Radial basis of |r|, unit vector harmonics up to order 3, and pair type."""
    r = rel.norm(dim=1, keepdim=True)
    mu = torch.linspace(0.5, cut, n_rbf, device=rel.device)
    rbf = torch.exp(-((r - mu) ** 2) / (2 * (cut / n_rbf) ** 2))
    phi = torch.atan2(rel[:, 1], rel[:, 0])[:, None]
    harm = torch.cat([torch.cos(m * phi) for m in (1, 2, 3, 4)] +
                     [torch.sin(m * phi) for m in (1, 2, 3, 4)], 1)
    ti = types[edge_index[1]]
    tj = types[edge_index[0]]
    pair = torch.nn.functional.one_hot(ti * 2 + tj, 4).float()
    return torch.cat([rbf, harm, pair, rbf[:, :1] * 0 + r], 1)


class EdgeConv(nn.Module):
    def __init__(self, hidden, edim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(edim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.lin = nn.Linear(hidden, hidden)
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, e):
        src, dst = edge_index
        m = self.gate(e) * self.lin(h)[src]
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        deg = torch.zeros(h.shape[0], 1, device=h.device).index_add_(
            0, dst, torch.ones(dst.shape[0], 1, device=h.device))
        agg = agg / deg.clamp(min=1)
        return self.norm(h + self.upd(torch.cat([h, agg], 1)))


class LYSNet(nn.Module):
    def __init__(self, n_out=6, hidden=64, layers=6, edim=16 + 8 + 4 + 1):
        super().__init__()
        self.embed = nn.Linear(2, hidden)
        self.convs = nn.ModuleList([EdgeConv(hidden, edim) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))

    def forward(self, x, edge_index, e):
        h = self.embed(x)
        for c in self.convs:
            h = c(h, edge_index, e)
        return self.head(h)


def rotate(rel, k, n_theta):
    a = k * math.pi / n_theta
    c, s = math.cos(a), math.sin(a)
    R = torch.tensor([[c, -s], [s, c]], dtype=rel.dtype)
    return rel @ R.T


def predict(model, sample):
    x, ei, rel, types = sample["x"], sample["ei"], sample["rel"], sample["types"]
    with torch.no_grad():
        return model(x, ei, edge_features(rel, types, ei)).numpy()
