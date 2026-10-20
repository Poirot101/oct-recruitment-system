"""Numba kernels for the 2D binary Lennard-Jones glass of Bamer et al. (2023).

Model (Bamer et al. Sect. 8.1, Eqs. 92-94; Falk & Langer, PRE 57, 7192 (1998)):
small (S, type 0) and large (L, type 1) particles with
sigma_SS = 2 sin(pi/10), sigma_LL = 2 sin(pi/5), sigma_SL = 1,
eps_SL = 1, eps_SS = eps_LL = 1/2, N_L/N_S = (1 + sqrt 5)/4.
The potential is shifted-force at r_c = 2.5 sigma_ab so that U and dU/dr vanish
at the cutoff, which keeps athermal quasistatic (AQS) minimisations smooth.

Everything that runs inside a minimisation loop is compiled with numba,
including the FIRE minimiser itself, so a strain step costs no Python overhead.

Conventions
-----------
* Periodic cell is a square of side L under simple shear gamma (Lees-Edwards):
  an image shifted by +L in y is shifted by gamma*L in x.
* Stress is the virial (Cauchy) stress with tension positive:
  sigma_ab = -(1/A) sum_{i<j} r_ij^a f_ij^b, where f_ij is the force on i from j
  and r_ij = r_i - r_j. Under shear x += dgamma*y the stress sigma_xy rises.
"""

import math

import numpy as np
from numba import njit, prange

SIGMA = np.array([[2.0 * math.sin(math.pi / 10.0), 1.0],
                  [1.0, 2.0 * math.sin(math.pi / 5.0)]])
EPS = np.array([[0.5, 1.0], [1.0, 0.5]])
RC = 2.5 * SIGMA
RCMAX = float(RC.max())
RATIO_L_TO_S = (1.0 + math.sqrt(5.0)) / 4.0


def _shift_tables():
    us = np.zeros((2, 2))
    dus = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            s, e, rc = SIGMA[a, b], EPS[a, b], RC[a, b]
            sr6 = (s / rc) ** 6
            us[a, b] = 4.0 * e * (sr6 * sr6 - sr6)
            dus[a, b] = 24.0 * e / rc * (sr6 - 2.0 * sr6 * sr6)
    return us, dus


USHIFT, DUSHIFT = _shift_tables()
RC2 = RC * RC


@njit(cache=True, fastmath=True, inline="always")
def pair_uf(r2, a, b, sig, eps, rc, us, dus):
    """Return (u, fs) with force on i = fs * r_ij (fs = -U'(r)/r)."""
    r = math.sqrt(r2)
    inv2 = sig[a, b] * sig[a, b] / r2
    sr6 = inv2 * inv2 * inv2
    sr12 = sr6 * sr6
    e = eps[a, b]
    u = 4.0 * e * (sr12 - sr6) - us[a, b] - (r - rc[a, b]) * dus[a, b]
    du = 24.0 * e / r * (sr6 - 2.0 * sr12) - dus[a, b]
    return u, -du / r


@njit(cache=True, fastmath=True, inline="always")
def min_image(dx, dy, L, gL):
    ny = np.rint(dy / L)
    dy -= ny * L
    dx -= ny * gL
    dx -= L * np.rint(dx / L)
    return dx, dy


# ---------------------------------------------------------------------------
# Periodic system: neighbour list, forces, FIRE
# ---------------------------------------------------------------------------

@njit(cache=True)
def build_nlist(pos, L, gL, rlist):
    n = pos.shape[0]
    rl2 = rlist * rlist
    cap = n * 40
    pi = np.empty(cap, np.int64)
    pj = np.empty(cap, np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = min_image(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], L, gL)
            if dx * dx + dy * dy < rl2:
                if k == cap:
                    cap *= 2
                    pi2 = np.empty(cap, np.int64)
                    pj2 = np.empty(cap, np.int64)
                    pi2[:k] = pi[:k]
                    pj2[:k] = pj[:k]
                    pi, pj = pi2, pj2
                pi[k] = i
                pj[k] = j
                k += 1
    return pi[:k].copy(), pj[:k].copy()


@njit(cache=True, fastmath=True)
def forces_pbc(pos, types, pi, pj, L, gL, sig, eps, rc, rc2, us, dus, f):
    """Fill f, return (energy, sxx, syy, sxy) with stresses divided by area."""
    f[:] = 0.0
    e = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for k in range(pi.shape[0]):
        i = pi[k]
        j = pj[k]
        a = types[i]
        b = types[j]
        dx, dy = min_image(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], L, gL)
        r2 = dx * dx + dy * dy
        if r2 < rc2[a, b]:
            u, fs = pair_uf(r2, a, b, sig, eps, rc, us, dus)
            e += u
            fx = fs * dx
            fy = fs * dy
            f[i, 0] += fx
            f[i, 1] += fy
            f[j, 0] -= fx
            f[j, 1] -= fy
            sxx += dx * fx
            syy += dy * fy
            sxy += 0.5 * (dx * fy + dy * fx)
    A = L * L
    return e, -sxx / A, -syy / A, -sxy / A


@njit(cache=True, fastmath=True)
def fire_pbc(pos, types, L, gL, pi, pj, ref, skin, ftol, maxit,
             sig, eps, rc, rc2, us, dus):
    """FIRE (Bitzek et al. 2006) on the periodic system, in place.

    `ref` holds positions at the last neighbour-list build; if any atom moves
    more than skin/2 from it, the list is rebuilt. Returns
    (pi, pj, iterations, force_calls, energy, sxy, fmax).
    """
    n = pos.shape[0]
    f = np.zeros((n, 2))
    v = np.zeros((n, 2))
    dt = 0.005
    dtmax = 0.05
    alpha = 0.1
    npos = 0
    calls = 0
    half2 = 0.25 * skin * skin
    rlist = RCMAX + skin
    e, sxx, syy, sxy = forces_pbc(pos, types, pi, pj, L, gL, sig, eps, rc, rc2, us, dus, f)
    calls += 1
    it = 0
    fmax = 0.0
    while it < maxit:
        fmax = 0.0
        for i in range(n):
            ff = f[i, 0] * f[i, 0] + f[i, 1] * f[i, 1]
            if ff > fmax:
                fmax = ff
        fmax = math.sqrt(fmax)
        if fmax < ftol:
            break
        P = 0.0
        vn = 0.0
        fn = 0.0
        for i in range(n):
            P += f[i, 0] * v[i, 0] + f[i, 1] * v[i, 1]
            vn += v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1]
            fn += f[i, 0] * f[i, 0] + f[i, 1] * f[i, 1]
        vn = math.sqrt(vn)
        fn = math.sqrt(fn)
        if P > 0.0:
            for i in range(n):
                for d in range(2):
                    v[i, d] = (1.0 - alpha) * v[i, d] + alpha * vn * f[i, d] / fn
            npos += 1
            if npos > 5:
                dt = min(dt * 1.1, dtmax)
                alpha *= 0.99
        else:
            npos = 0
            dt *= 0.5
            alpha = 0.1
            for i in range(n):
                pos[i, 0] -= 0.5 * dt * v[i, 0]
                pos[i, 1] -= 0.5 * dt * v[i, 1]
            v[:] = 0.0
        # semi-implicit Euler (MD step)
        for i in range(n):
            v[i, 0] += dt * f[i, 0]
            v[i, 1] += dt * f[i, 1]
            pos[i, 0] += dt * v[i, 0]
            pos[i, 1] += dt * v[i, 1]
        # neighbour list check (affine shear is applied outside, so ref is
        # updated consistently by the caller)
        rebuild = False
        for i in range(n):
            dx = pos[i, 0] - ref[i, 0]
            dy = pos[i, 1] - ref[i, 1]
            if dx * dx + dy * dy > half2:
                rebuild = True
                break
        if rebuild:
            pi, pj = build_nlist(pos, L, gL, rlist)
            ref[:] = pos
        e, sxx, syy, sxy = forces_pbc(pos, types, pi, pj, L, gL, sig, eps, rc, rc2, us, dus, f)
        calls += 1
        it += 1
    return pi, pj, it, calls, e, sxy, fmax


@njit(cache=True, fastmath=True)
def md_quench(pos, types, L, T0, T1, nsteps, dt, tau, seed, sig, eps, rc, rc2, us, dus, skin):
    """Velocity-Verlet MD with Berendsen rescaling, T ramps from T0 to T1."""
    np.random.seed(seed)
    n = pos.shape[0]
    v = np.random.normal(0.0, math.sqrt(T0), (n, 2))
    for d in range(2):
        v[:, d] -= v[:, d].mean()
    f = np.zeros((n, 2))
    rlist = RCMAX + skin
    pi, pj = build_nlist(pos, L, 0.0, rlist)
    ref = pos.copy()
    half2 = 0.25 * skin * skin
    forces_pbc(pos, types, pi, pj, L, 0.0, sig, eps, rc, rc2, us, dus, f)
    for s in range(nsteps):
        T = T0 + (T1 - T0) * s / max(nsteps - 1, 1)
        for i in range(n):
            v[i, 0] += 0.5 * dt * f[i, 0]
            v[i, 1] += 0.5 * dt * f[i, 1]
            pos[i, 0] += dt * v[i, 0]
            pos[i, 1] += dt * v[i, 1]
        rebuild = False
        for i in range(n):
            dx = pos[i, 0] - ref[i, 0]
            dy = pos[i, 1] - ref[i, 1]
            if dx * dx + dy * dy > half2:
                rebuild = True
                break
        if rebuild:
            for i in range(n):
                pos[i, 0] -= L * math.floor(pos[i, 0] / L)
                pos[i, 1] -= L * math.floor(pos[i, 1] / L)
            pi, pj = build_nlist(pos, L, 0.0, rlist)
            ref[:] = pos
        forces_pbc(pos, types, pi, pj, L, 0.0, sig, eps, rc, rc2, us, dus, f)
        ke = 0.0
        for i in range(n):
            v[i, 0] += 0.5 * dt * f[i, 0]
            v[i, 1] += 0.5 * dt * f[i, 1]
            ke += v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1]
        Tk = ke / (2.0 * (n - 1))
        lam = math.sqrt(max(1.0 + dt / tau * (T / max(Tk, 1e-12) - 1.0), 0.0))
        for i in range(n):
            v[i, 0] *= lam
            v[i, 1] *= lam
    for i in range(n):
        pos[i, 0] -= L * math.floor(pos[i, 0] / L)
        pos[i, 1] -= L * math.floor(pos[i, 1] / L)


@njit(cache=True, fastmath=True)
def per_atom_energy(pos, types, pi, pj, L, gL, sig, eps, rc, rc2, us, dus):
    n = pos.shape[0]
    ea = np.zeros(n)
    for k in range(pi.shape[0]):
        i = pi[k]
        j = pj[k]
        a = types[i]
        b = types[j]
        dx, dy = min_image(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], L, gL)
        r2 = dx * dx + dy * dy
        if r2 < rc2[a, b]:
            u, fs = pair_uf(r2, a, b, sig, eps, rc, us, dus)
            ea[i] += 0.5 * u
            ea[j] += 0.5 * u
    return ea


# ---------------------------------------------------------------------------
# Local yield stress (Patinet et al. 2016; Barbot et al. 2018)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def cluster_forces(x, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, f, tx, ty, nx, ny, area):
    """Forces on a non-periodic cluster; atoms [0, nfree) are free.

    Returns (energy, tau) where tau is the shear stress of the free region
    resolved on the loading system (t, n): tau = sym(t . sigma . n).
    """
    f[:] = 0.0
    e = 0.0
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for k in range(ci.shape[0]):
        i = ci[k]
        j = cj[k]
        a = types[i]
        b = types[j]
        dx = x[i, 0] - x[j, 0]
        dy = x[i, 1] - x[j, 1]
        r2 = dx * dx + dy * dy
        if r2 < rc2[a, b]:
            u, fs = pair_uf(r2, a, b, sig, eps, rc, us, dus)
            fx = fs * dx
            fy = fs * dy
            f[i, 0] += fx
            f[i, 1] += fy
            f[j, 0] -= fx
            f[j, 1] -= fy
            # a pair contributes half its virial to each free atom it touches
            w = 0.0
            if i < nfree:
                w += 0.5
            if j < nfree:
                w += 0.5
            e += w * u
            sxx += w * dx * fx
            syy += w * dy * fy
            sxy += w * 0.5 * (dx * fy + dy * fx)
    tau = -(tx * (sxx * nx + sxy * ny) + ty * (sxy * nx + syy * ny)) / area
    return e, tau


@njit(cache=True, fastmath=True)
def fire_cluster(x, types, nfree, ci, cj, ftol, maxit, sig, eps, rc, rc2, us, dus,
                 tx, ty, nx, ny, area):
    """FIRE on the free atoms of a cluster, in place. Returns (it, e, tau)."""
    n = x.shape[0]
    f = np.zeros((n, 2))
    v = np.zeros((nfree, 2))
    dt = 0.005
    dtmax = 0.05
    alpha = 0.1
    npos = 0
    e, tau = cluster_forces(x, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, f,
                            tx, ty, nx, ny, area)
    it = 0
    while it < maxit:
        fmax = 0.0
        P = 0.0
        vn = 0.0
        fn = 0.0
        for i in range(nfree):
            ff = f[i, 0] * f[i, 0] + f[i, 1] * f[i, 1]
            if ff > fmax:
                fmax = ff
            P += f[i, 0] * v[i, 0] + f[i, 1] * v[i, 1]
            vn += v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1]
            fn += ff
        if math.sqrt(fmax) < ftol:
            break
        vn = math.sqrt(vn)
        fn = math.sqrt(fn)
        if P > 0.0:
            for i in range(nfree):
                for d in range(2):
                    v[i, d] = (1.0 - alpha) * v[i, d] + alpha * vn * f[i, d] / fn
            npos += 1
            if npos > 5:
                dt = min(dt * 1.1, dtmax)
                alpha *= 0.99
        else:
            npos = 0
            dt *= 0.5
            alpha = 0.1
            for i in range(nfree):
                x[i, 0] -= 0.5 * dt * v[i, 0]
                x[i, 1] -= 0.5 * dt * v[i, 1]
            v[:] = 0.0
        for i in range(nfree):
            v[i, 0] += dt * f[i, 0]
            v[i, 1] += dt * f[i, 1]
            x[i, 0] += dt * v[i, 0]
            x[i, 1] += dt * v[i, 1]
        e, tau = cluster_forces(x, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, f,
                                tx, ty, nx, ny, area)
        it += 1
    return it, e, tau


@njit(cache=True)
def cluster_pairs(x, rlist):
    n = x.shape[0]
    rl2 = rlist * rlist
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            if dx * dx + dy * dy < rl2:
                cnt += 1
    ci = np.empty(cnt, np.int64)
    cj = np.empty(cnt, np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i, 0] - x[j, 0]
            dy = x[i, 1] - x[j, 1]
            if dx * dx + dy * dy < rl2:
                ci[k] = i
                cj[k] = j
                k += 1
    return ci, cj


@njit(cache=True, fastmath=True)
def lys_direction(X0, types, nfree, ci, cj, theta, mode, dg_min, dg_coarse, gmax,
                  jump_tol, slope_frac, ftol, sig, eps, rc, rc2, us, dus, area):
    """Load one cluster in simple shear along theta until the first instability.

    mode 0 (baseline): fixed steps dg_min, affine trial positions.
    mode 1 (accelerated): steps up to dg_coarse with a secant (linear-response)
        predictor for the non-affine motion; a coarse step is kept only if it
        passes an elasticity check, otherwise the last stable state is restored
        and the step is halved, down to dg_min; after each accepted step the
        step is doubled again (capped at dg_coarse).
    Both modes certify the instability on the same dg_min grid, so they
    return the same critical strain up to the grid resolution.

    Returns (tau0, tau_c, gamma_c, minimiser_iterations).
    """
    tx = math.cos(theta)
    ty = math.sin(theta)
    nx = -math.sin(theta)
    ny = math.cos(theta)
    n = X0.shape[0]
    # affine direction field: d x / d gamma = (n . X0) t
    ax = np.empty(n)
    ay = np.empty(n)
    for i in range(n):
        s = nx * X0[i, 0] + ny * X0[i, 1]
        ax[i] = s * tx
        ay[i] = s * ty
    x = X0.copy()
    it_tot = 0
    it, e, tau = fire_cluster(x, types, nfree, ci, cj, ftol, 100000, sig, eps, rc, rc2,
                              us, dus, tx, ty, nx, ny, area)
    it_tot += it
    tau0 = tau
    g = 0.0
    tau_prev = tau
    xprev = x.copy()
    # non-affine velocity estimate (free atoms) for the secant predictor
    vna = np.zeros((nfree, 2))
    have_v = False
    slope = 0.0
    have_slope = False
    dg = dg_min if mode == 0 else dg_coarse
    xt = np.empty_like(x)
    while g < gmax:
        # trial state
        for i in range(n):
            xt[i, 0] = xprev[i, 0] + dg * ax[i]
            xt[i, 1] = xprev[i, 1] + dg * ay[i]
        if mode == 1 and have_v:
            for i in range(nfree):
                xt[i, 0] += dg * vna[i, 0]
                xt[i, 1] += dg * vna[i, 1]
        x[:] = xt
        it, e, tau = fire_cluster(x, types, nfree, ci, cj, ftol, 100000, sig, eps, rc,
                                  rc2, us, dus, tx, ty, nx, ny, area)
        it_tot += it
        jump = 0.0
        for i in range(nfree):
            dx = x[i, 0] - xt[i, 0]
            dy = x[i, 1] - xt[i, 1]
            d2 = dx * dx + dy * dy
            if d2 > jump:
                jump = d2
        jump = math.sqrt(jump)
        fine = dg <= dg_min * 1.000001
        if fine:
            # the certification test, identical in both modes
            event = (tau < tau_prev) or (jump > jump_tol)
        else:
            # a coarse step is accepted only if it looks purely elastic: the
            # stress rose by at least half the elastic expectation and the
            # relaxed state stayed close to the predictor
            if have_slope:
                event = tau < tau_prev + slope_frac * slope * dg
            else:
                event = tau < tau_prev
            event = event or (jump > 0.2 * jump_tol)
        if event:
            if fine:
                return tau0, tau_prev, g, it_tot
            dg = max(0.5 * dg, dg_min)
            continue
        # accept
        if mode == 1:
            for i in range(nfree):
                vna[i, 0] = (x[i, 0] - xprev[i, 0]) / dg - ax[i]
                vna[i, 1] = (x[i, 1] - xprev[i, 1]) / dg - ay[i]
            have_v = True
            slope = (tau - tau_prev) / dg
            have_slope = True
        xprev[:] = x
        tau_prev = tau
        g += dg
        if mode == 1:
            dg = min(2.0 * dg, dg_coarse)
    return tau0, tau_prev, g, it_tot


@njit(cache=True)
def extract_cluster(pos, types, L, center, r_free, r_tot):
    """Atoms within r_tot of `center` (minimum image, unsheared cell),
    centred on it, free atoms (r < r_free) first. Returns (X, typ, nfree, idx)."""
    n = pos.shape[0]
    cx = pos[center, 0]
    cy = pos[center, 1]
    rel = np.empty((n, 2))
    r2 = np.empty(n)
    for i in range(n):
        dx, dy = min_image(pos[i, 0] - cx, pos[i, 1] - cy, L, 0.0)
        rel[i, 0] = dx
        rel[i, 1] = dy
        r2[i] = dx * dx + dy * dy
    free = np.where(r2 < r_free * r_free)[0]
    shell = np.where((r2 >= r_free * r_free) & (r2 < r_tot * r_tot))[0]
    idx = np.concatenate((free, shell))
    return rel[idx].copy(), types[idx].copy(), free.shape[0], idx


@njit(cache=True, parallel=True)
def lys_all(pos, types, L, centers, thetas, r_free, mode, dg_min, dg_coarse, gmax,
            jump_tol, slope_frac, ftol, sig, eps, rc, rc2, us, dus):
    """Local yield stress for every centre x direction (parallel over centres).

    Returns arrays (tau0, tau_c, gamma_c, iterations) of shape (n_centres, n_theta).
    """
    nc = centers.shape[0]
    nt = thetas.shape[0]
    tau0 = np.zeros((nc, nt))
    tauc = np.zeros((nc, nt))
    gc = np.zeros((nc, nt))
    its = np.zeros((nc, nt), np.int64)
    r_tot = r_free + RCMAX
    area = math.pi * r_free * r_free
    for k in prange(nc):
        X, typ, nfree, idx = extract_cluster(pos, types, L, centers[k], r_free, r_tot)
        ci, cj = cluster_pairs(X, RCMAX + 1.2)
        for m in range(nt):
            a, b, c, d = lys_direction(X, typ, nfree, ci, cj, thetas[m], mode, dg_min,
                                       dg_coarse, gmax, jump_tol, slope_frac, ftol, sig, eps,
                                       rc, rc2, us, dus, area)
            tau0[k, m] = a
            tauc[k, m] = b
            gc[k, m] = c
            its[k, m] = d
    return tau0, tauc, gc, its


# ---------------------------------------------------------------------------
# Soft-mode reduced-order predictor of the local yield strain
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def cluster_hessian(x, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus):
    """Dense Hessian of the cluster energy w.r.t. the free-atom coordinates."""
    H = np.zeros((2 * nfree, 2 * nfree))
    for k in range(ci.shape[0]):
        i = ci[k]
        j = cj[k]
        if i >= nfree and j >= nfree:
            continue
        a = types[i]
        b = types[j]
        dx = x[i, 0] - x[j, 0]
        dy = x[i, 1] - x[j, 1]
        r2 = dx * dx + dy * dy
        if r2 >= rc2[a, b]:
            continue
        r = math.sqrt(r2)
        s2 = sig[a, b] * sig[a, b] / r2
        sr6 = s2 * s2 * s2
        sr12 = sr6 * sr6
        e = eps[a, b]
        du = 24.0 * e / r * (sr6 - 2.0 * sr12) - dus[a, b]
        d2u = 24.0 * e / r2 * (26.0 * sr12 - 7.0 * sr6)
        ux = dx / r
        uy = dy / r
        u = (ux, uy)
        for p in range(2):
            for q in range(2):
                kpq = (d2u - du / r) * u[p] * u[q]
                if p == q:
                    kpq += du / r
                if i < nfree:
                    H[2 * i + p, 2 * i + q] += kpq
                if j < nfree:
                    H[2 * j + p, 2 * j + q] += kpq
                if i < nfree and j < nfree:
                    H[2 * i + p, 2 * j + q] -= kpq
                    H[2 * j + p, 2 * i + q] -= kpq
    return H


@njit(cache=True, fastmath=True)
def _free_force_dot(x, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, phi, f):
    cluster_forces(x, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, f, 1.0, 0.0, 0.0, 1.0, 1.0)
    s = 0.0
    for i in range(nfree):
        s += phi[2 * i] * f[i, 0] + phi[2 * i + 1] * f[i, 1]
    return s


@njit(cache=True, fastmath=True)
def softmode_yield(X0, types, nfree, ci, cj, thetas, n_modes, sig, eps, rc, rc2, us, dus):
    """Cubic soft-mode estimate of the yield strain for each direction.

    Along each of the n_modes softest modes phi_k (eigenvalue lambda_k) of the
    frozen-boundary cluster, the energy is expanded to third order,
    E(q) = lambda q^2/2 + kappa q^3/6, and simple shear along theta adds the
    generalised force gamma*h_k with h_k = phi_k . dF/dgamma. The mode loses
    stability (saddle-node, Maloney & Lemaitre 2006) at
    gamma_k = -lambda_k^2 / (2 kappa_k h_k) when that is positive. The estimate
    is the minimum over modes. Returns (gamma_pred [n_theta], lambda_min).
    """
    n = X0.shape[0]
    H = cluster_hessian(X0, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus)
    w, V = np.linalg.eigh(H)
    f = np.zeros((n, 2))
    xs = X0.copy()
    m = min(n_modes, 2 * nfree)
    kappa = np.zeros(m)
    s = 1e-3
    for k in range(m):
        phi = V[:, k].copy()
        g = np.zeros(3)
        for c in range(3):
            step = (c - 1) * s
            for i in range(nfree):
                xs[i, 0] = X0[i, 0] + step * phi[2 * i]
                xs[i, 1] = X0[i, 1] + step * phi[2 * i + 1]
            g[c] = _free_force_dot(xs, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, phi, f)
        kappa[k] = -(g[2] - 2.0 * g[1] + g[0]) / (s * s)
    nt = thetas.shape[0]
    out = np.full(nt, np.inf)
    d = 1e-6
    fp = np.zeros((n, 2))
    fm = np.zeros((n, 2))
    xa = X0.copy()
    for t in range(nt):
        tx = math.cos(thetas[t])
        ty = math.sin(thetas[t])
        nx = -math.sin(thetas[t])
        ny = math.cos(thetas[t])
        for sgn in range(2):
            dd = d if sgn == 0 else -d
            for i in range(n):
                sa = nx * X0[i, 0] + ny * X0[i, 1]
                xa[i, 0] = X0[i, 0] + dd * sa * tx
                xa[i, 1] = X0[i, 1] + dd * sa * ty
            if sgn == 0:
                cluster_forces(xa, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, fp,
                               1.0, 0.0, 0.0, 1.0, 1.0)
            else:
                cluster_forces(xa, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, fm,
                               1.0, 0.0, 0.0, 1.0, 1.0)
        for k in range(m):
            h = 0.0
            for i in range(nfree):
                h += V[2 * i, k] * (fp[i, 0] - fm[i, 0]) + V[2 * i + 1, k] * (fp[i, 1] - fm[i, 1])
            h /= 2.0 * d
            den = 2.0 * kappa[k] * h
            if den < 0.0:
                gk = -w[k] * w[k] / den
                if gk < out[t]:
                    out[t] = gk
    return out, w[0]


@njit(cache=True, parallel=True)
def softmode_all(pos, types, L, centers, thetas, r_free, n_modes, sig, eps, rc, rc2, us, dus):
    nc = centers.shape[0]
    nt = thetas.shape[0]
    gp = np.zeros((nc, nt))
    lmin = np.zeros(nc)
    r_tot = r_free + RCMAX
    for k in prange(nc):
        X, typ, nfree, idx = extract_cluster(pos, types, L, centers[k], r_free, r_tot)
        ci, cj = cluster_pairs(X, RCMAX + 1.2)
        g, l0 = softmode_yield(X, typ, nfree, ci, cj, thetas, n_modes, sig, eps, rc, rc2,
                               us, dus)
        gp[k] = g
        lmin[k] = l0
    return gp, lmin


@njit(cache=True, fastmath=True)
def softmode_terms(X0, types, nfree, ci, cj, thetas, n_modes, sig, eps, rc, rc2, us, dus):
    """Return (lambda[m], kappa[m], h[n_theta, m]) of the softmode expansion."""
    n = X0.shape[0]
    H = cluster_hessian(X0, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus)
    w, V = np.linalg.eigh(H)
    f = np.zeros((n, 2))
    xs = X0.copy()
    m = min(n_modes, 2 * nfree)
    kappa = np.zeros(m)
    s = 1e-3
    for k in range(m):
        phi = V[:, k].copy()
        g = np.zeros(3)
        for c in range(3):
            step = (c - 1) * s
            for i in range(nfree):
                xs[i, 0] = X0[i, 0] + step * phi[2 * i]
                xs[i, 1] = X0[i, 1] + step * phi[2 * i + 1]
            g[c] = _free_force_dot(xs, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, phi, f)
        kappa[k] = -(g[2] - 2.0 * g[1] + g[0]) / (s * s)
    nt = thetas.shape[0]
    hh = np.zeros((nt, m))
    d = 1e-6
    fp = np.zeros((n, 2))
    fm = np.zeros((n, 2))
    xa = X0.copy()
    for t in range(nt):
        tx = math.cos(thetas[t])
        ty = math.sin(thetas[t])
        nx = -math.sin(thetas[t])
        ny = math.cos(thetas[t])
        for i in range(n):
            sa = nx * X0[i, 0] + ny * X0[i, 1]
            xa[i, 0] = X0[i, 0] + d * sa * tx
            xa[i, 1] = X0[i, 1] + d * sa * ty
        cluster_forces(xa, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, fp, 1.0, 0.0, 0.0, 1.0, 1.0)
        for i in range(n):
            sa = nx * X0[i, 0] + ny * X0[i, 1]
            xa[i, 0] = X0[i, 0] - d * sa * tx
            xa[i, 1] = X0[i, 1] - d * sa * ty
        cluster_forces(xa, types, nfree, ci, cj, sig, eps, rc, rc2, us, dus, fm, 1.0, 0.0, 0.0, 1.0, 1.0)
        for k in range(m):
            h = 0.0
            for i in range(nfree):
                h += V[2 * i, k] * (fp[i, 0] - fm[i, 0]) + V[2 * i + 1, k] * (fp[i, 1] - fm[i, 1])
            hh[t, k] = h / (2.0 * d)
    return w[:m].copy(), kappa, hh


@njit(cache=True, parallel=True)
def softmode_terms_all(pos, types, L, centers, thetas, r_free, n_modes, sig, eps, rc, rc2, us, dus):
    nc = centers.shape[0]
    nt = thetas.shape[0]
    W = np.zeros((nc, n_modes))
    K = np.zeros((nc, n_modes))
    Hh = np.zeros((nc, nt, n_modes))
    r_tot = r_free + RCMAX
    for k in prange(nc):
        X, typ, nfree, idx = extract_cluster(pos, types, L, centers[k], r_free, r_tot)
        ci, cj = cluster_pairs(X, RCMAX + 1.2)
        w, ka, hh = softmode_terms(X, typ, nfree, ci, cj, thetas, n_modes, sig, eps, rc, rc2,
                                   us, dus)
        W[k] = w
        K[k] = ka
        Hh[k] = hh
    return W, K, Hh
