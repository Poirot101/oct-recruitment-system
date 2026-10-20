"""Python-level drivers: glass generation, global AQS, local yield stress maps."""

import time

import numpy as np

from . import sim

P = (sim.SIGMA, sim.EPS, sim.RC, sim.RC2, sim.USHIFT, sim.DUSHIFT)
SKIN = 0.4


def random_start(n, density, seed):
    rng = np.random.default_rng(seed)
    n_l = int(round(n * sim.RATIO_L_TO_S / (1.0 + sim.RATIO_L_TO_S)))
    types = np.zeros(n, np.int64)
    types[rng.choice(n, n_l, replace=False)] = 1
    L = float(np.sqrt(n / density))
    m = int(np.ceil(np.sqrt(n)))
    a = L / m
    grid = np.stack(np.meshgrid(np.arange(m), np.arange(m), indexing="ij"), -1).reshape(-1, 2)
    grid = grid[rng.choice(m * m, n, replace=False)]
    pos = (grid + 0.5 + 0.1 * (rng.random((n, 2)) - 0.5)) * a
    return np.ascontiguousarray(pos, dtype=np.float64), types, L


def minimize(pos, types, L, gL=0.0, ftol=1e-10, maxit=2_000_000):
    pi, pj = sim.build_nlist(pos, L, gL, sim.RCMAX + SKIN)
    ref = pos.copy()
    pi, pj, it, calls, e, sxy, fmax = sim.fire_pbc(pos, types, L, gL, pi, pj, ref, SKIN, ftol,
                                                   maxit, *P)
    return dict(iterations=it, energy=e / len(pos), sxy=sxy, fmax=fmax)


def make_glass(n=1024, density=1.2, T_melt=1.0, T_end=0.05, n_equil=20000, n_quench=20000,
               dt=0.005, seed=0):
    """Melt-quench a glass (Bamer et al. Sect. 8.1) and return an inherent structure."""
    pos, types, L = random_start(n, density, seed)
    minimize(pos, types, L, ftol=1e-2, maxit=20000)  # remove overlaps
    sim.md_quench(pos, types, L, T_melt, T_melt, n_equil, dt, 0.1, seed, *P, SKIN)
    sim.md_quench(pos, types, L, T_melt, T_end, n_quench, dt, 0.1, seed + 1, *P, SKIN)
    info = minimize(pos, types, L)
    pi, pj = sim.build_nlist(pos, L, 0.0, sim.RCMAX + SKIN)
    f = np.zeros_like(pos)
    e, sxx, syy, sxy = sim.forces_pbc(pos, types, pi, pj, L, 0.0, *P, f)
    info.update(pressure=-(sxx + syy) / 2.0, L=L)
    return pos, types, L, info


def global_aqs(pos, types, L, dgamma=1e-4, gmax=0.12, ftol=1e-10, drop_tol=1e-4):
    """Athermal quasistatic simple shear of the periodic glass (Bamer Sect. 9.2).

    Returns dict with the stress-strain curve and the list of plastic events.
    An event is a stress drop larger than drop_tol; its location is the atom
    with the largest non-affine displacement over that step.
    """
    pos = pos.copy()
    n = len(pos)
    gL = 0.0
    pi, pj = sim.build_nlist(pos, L, gL, sim.RCMAX + SKIN)
    ref = pos.copy()
    pi, pj, it, calls, e, sxy, fmax = sim.fire_pbc(pos, types, L, gL, pi, pj, ref, SKIN,
                                                   ftol, 2_000_000, *P)
    gam, stress, events = [0.0], [sxy], []
    g = 0.0
    total_it = it
    while g < gmax - 1e-12:
        prev = pos.copy()
        pos[:, 0] += dgamma * pos[:, 1]
        ref[:, 0] += dgamma * ref[:, 1]
        gL += dgamma * L
        g += dgamma
        trial = pos.copy()
        pi, pj, it, calls, e, s_new, fmax = sim.fire_pbc(pos, types, L, gL, pi, pj, ref, SKIN,
                                                         ftol, 2_000_000, *P)
        total_it += it
        if s_new < stress[-1] - drop_tol:
            d = pos - trial
            d[:, 0] -= L * np.rint(d[:, 0] / L)
            d[:, 1] -= L * np.rint(d[:, 1] / L)
            na = np.sqrt((d ** 2).sum(1))
            events.append(dict(gamma=g, atom=int(np.argmax(na)), drop=stress[-1] - s_new,
                               nonaffine=na.astype(np.float32)))
        gam.append(g)
        stress.append(s_new)
    return dict(gamma=np.array(gam), stress=np.array(stress), events=events,
                iterations=total_it)


def lys_map(pos, types, L, centers=None, n_theta=6, r_free=5.0, mode=1, dg_min=1e-4,
            dg_coarse=2e-3, gmax=0.25, jump_tol=0.1, slope_frac=0.8, ftol=1e-9):
    if centers is None:
        centers = np.arange(len(pos))
    centers = np.asarray(centers, np.int64)
    thetas = np.arange(n_theta) * np.pi / n_theta
    t0 = time.perf_counter()
    tau0, tauc, gc, its = sim.lys_all(pos, types, L, centers, thetas, r_free, mode, dg_min,
                                      dg_coarse, gmax, jump_tol, slope_frac, ftol, *P)
    wall = time.perf_counter() - t0
    return dict(centers=centers, thetas=thetas, tau0=tau0, tauc=tauc, gammac=gc,
                iterations=its, wall=wall)
