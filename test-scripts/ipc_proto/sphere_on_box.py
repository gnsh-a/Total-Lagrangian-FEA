#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "ipctk==1.6.0"]
# ///
"""
IPC prototype: a T10 sphere hitting a box, solved with the same velocity-form
backward-Euler Newton that SyncedNewton uses, plus an IPC barrier.

SyncedNewton today (lib_src/solvers/SyncedNewton.cu, solver_grad_L):
    R(v) = (M/h)(v - v_n) + f_int(x) - f_ext + h J^T (lam + rho c)
    H(v) = M/h + h K_t(x) + h^2 rho J^T J,          x = x_n + h v
with contact forces computed once per step and frozen inside f_ext.

This prototype adds contact as one more energy term:
    Phi(v) = 1/(2h) |v - v_n|_M^2 + (Psi(x) - f_ext.x + kappa B(x)) / h + D(v)
    R(v)   = (M/h)(v - v_n) + f_int(x) - f_ext + kappa grad B(x) + grad D(v)
    H(v)   = M/h + h (K_t(x) + kappa hess B(x)) + hess D(v)
and replaces the plain Newton update with a CCD-capped Armijo search on Phi.

Contact modes
    ipc      implicit barrier, evaluated at every Newton iterate
    penalty  explicit lagged penalty (how DEME/hydro forces enter today)

Scenes
    rigid    T10 sphere vs analytic rigid box (barrier, distance and CCD in this file)
    fem      T10 sphere vs T10 box with a fixed bottom (mesh-mesh IPC through ipctk)

Examples
    uv run test-scripts/ipc_proto/sphere_on_box.py --scene rigid --contact ipc --dt 1e-3
    uv run test-scripts/ipc_proto/sphere_on_box.py --scene rigid --contact penalty --dt 1e-3
    uv run test-scripts/ipc_proto/sphere_on_box.py --scene fem --dt 1e-3
    uv run test-scripts/ipc_proto/sphere_on_box.py --suite   # all runs used in the write-up
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
I3 = np.eye(3)

# Repo T10 edge order (FEAT10Data.cu): nodes 4..9 sit on these corner pairs.
EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))  # face opposite corner k


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def standardize_t10(X, T):
    """Positive orientation and repo edge order, assigning midside nodes by geometry."""
    T = T.copy()
    C = X[T[:, :4]]
    vol = np.einsum("ei,ei->e", np.cross(C[:, 1] - C[:, 0], C[:, 2] - C[:, 0]), C[:, 3] - C[:, 0])
    neg = vol < 0
    T[neg, 1], T[neg, 2] = T[neg, 2].copy(), T[neg, 1].copy()
    out = T.copy()
    mids = X[T[:, 4:]]
    rows = np.arange(len(T))
    for k, (i, j) in enumerate(EDGES):
        target = 0.5 * (X[T[:, i]] + X[T[:, j]])
        out[:, 4 + k] = T[rows, 4 + np.linalg.norm(mids - target[:, None], axis=2).argmin(1)]
    return out


def read_tetgen_t10(prefix):
    def rows(path):
        return [l.split() for l in open(path) if l.strip() and not l.lstrip().startswith("#")]

    nr, er = rows(prefix + ".node"), rows(prefix + ".ele")
    n, m = int(nr[0][0]), int(er[0][0])
    ids = np.array([int(r[0]) for r in nr[1 : n + 1]])
    base = ids.min()
    X = np.empty((n, 3))
    X[ids - base] = [[float(v) for v in r[1:4]] for r in nr[1 : n + 1]]
    T = np.array([[int(v) for v in r[1:11]] for r in er[1 : m + 1]]) - base
    return X, standardize_t10(X, T)


def box_t10(lo, hi, n):
    """Structured T10 box: each hex split into 6 Kuhn tets, midside nodes on straight edges."""
    nx, ny, nz = n
    g = [np.linspace(lo[a], hi[a], n[a] + 1) for a in range(3)]
    vid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    P = np.array([[g[0][i], g[1][j], g[2][k]] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)])
    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        for perm in itertools.permutations(range(3)):
            c, path = [0, 0, 0], [vid(i, j, k)]
            for ax in perm:
                c[ax] += 1
                path.append(vid(i + c[0], j + c[1], k + c[2]))
            tets.append(path)
    X, T, edge_id = list(P), np.zeros((len(tets), 10), int), {}
    for e, t in enumerate(tets):
        T[e, :4] = t
        for k, (a, b) in enumerate(EDGES):
            key = (min(t[a], t[b]), max(t[a], t[b]))
            if key not in edge_id:
                edge_id[key] = len(X)
                X.append(0.5 * (P[t[a]] + P[t[b]]))
            T[e, 4 + k] = edge_id[key]
    X = np.array(X)
    return X, standardize_t10(X, T)


def surface_triangles_4way(T):
    """Boundary faces of a T10 mesh, each split into 4 linear triangles on its 6 nodes.

    This is the constant-W proxy from the write-up: collision vertices are FE nodes,
    so the displacement map W is a selection matrix (lib_utils/surface_trimesh_extract.cc
    builds the same split)."""
    edge_slot = {frozenset(e): 4 + k for k, e in enumerate(EDGES)}
    count = {}
    for e, t in enumerate(T):
        for f in FACES:
            key = tuple(sorted(t[list(f)]))
            count.setdefault(key, []).append((e, f))
    tris = []
    for key, owners in count.items():
        if len(owners) != 1:
            continue
        e, f = owners[0]
        t = T[e]
        c0, c1, c2 = (t[i] for i in f)
        m01 = t[edge_slot[frozenset((f[0], f[1]))]]
        m12 = t[edge_slot[frozenset((f[1], f[2]))]]
        m02 = t[edge_slot[frozenset((f[0], f[2]))]]
        tris += [(c0, m01, m02), (m01, c1, m12), (m02, m12, c2), (m01, m12, m02)]
    return np.array(tris, dtype=np.int32)


def vertex_areas(X, tris, n_nodes):
    a = 0.5 * np.linalg.norm(np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]]), axis=1)
    A = np.zeros(n_nodes)
    for k in range(3):
        np.add.at(A, tris[:, k], a / 3.0)
    return A


# ---------------------------------------------------------------------------
# T10 shape functions and quadrature
# ---------------------------------------------------------------------------
def t10_shape(xi):
    l = np.column_stack([1.0 - xi.sum(1), xi])
    N = np.empty((len(xi), 10))
    for i in range(4):
        N[:, i] = l[:, i] * (2 * l[:, i] - 1)
    for k, (i, j) in enumerate(EDGES):
        N[:, 4 + k] = 4 * l[:, i] * l[:, j]
    return N


def t10_shape_grads(xi):
    l = np.column_stack([1.0 - xi.sum(1), xi])
    dl = np.array([[-1.0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dN = np.empty((len(xi), 10, 3))
    for i in range(4):
        dN[:, i] = (4 * l[:, i] - 1)[:, None] * dl[i]
    for k, (i, j) in enumerate(EDGES):
        dN[:, 4 + k] = 4 * (l[:, i, None] * dl[j] + l[:, j, None] * dl[i])
    return dN


def tet_rule_4pt():
    """Degree-2 rule, positive weights (stiffness)."""
    a, b = 0.5854101966249685, 0.1381966011250105
    pts = np.array([[b, b, b], [a, b, b], [b, a, b], [b, b, a]])
    return pts, np.full(4, 1.0 / 24.0)


def tet_rule_collapsed(n=4):
    """Collapsed (Duffy) Gauss rule, positive weights, exact to degree 2n-3 (mass: degree 4).

    The repo's 5-point Keast rule has a negative centroid weight and makes the T10
    consistent mass indefinite; this rule does not."""
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = 0.5 * (g + 1), 0.5 * w
    pts, wts = [], []
    for (u, wu), (v, wv), (s, ws) in itertools.product(zip(g, w), zip(g, w), zip(g, w)):
        pts.append((u, (1 - u) * v, (1 - u) * (1 - v) * s))
        wts.append(wu * wv * ws * (1 - u) ** 2 * (1 - v))
    return np.array(pts), np.array(wts)


def psd_project(Hb):
    """Clamp negative eigenvalues of a batch of symmetric blocks."""
    w, Q = np.linalg.eigh(Hb)
    return np.einsum("...ij,...j,...kj->...ik", Q, np.maximum(w, 0.0), Q)


# ---------------------------------------------------------------------------
# FE model: T10, St. Venant-Kirchhoff (same constitutive law as the repo demos)
# ---------------------------------------------------------------------------
class T10Model:
    def __init__(self, X, T, E, nu, rho, fixed_nodes=(), project_elastic=True):
        self.X0, self.T = X, T
        self.n = len(X)
        self.ndof = 3 * self.n
        self.project_elastic = project_elastic
        ne = len(T)
        E, nu, rho = (np.broadcast_to(np.asarray(a, float), (ne,)) for a in (E, nu, rho))
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        xq, wq = tet_rule_4pt()
        dN = t10_shape_grads(xq)
        J = np.einsum("eai,qaj->eqij", X[T], dN)
        self.gradN = np.einsum("qaj,eqjk->eqak", dN, np.linalg.inv(J))
        self.dV = np.linalg.det(J) * wq
        assert (self.dV > 0).all(), "inverted element"

        xm, wm = tet_rule_collapsed(4)
        Nm, dNm = t10_shape(xm), t10_shape_grads(xm)
        detJm = np.linalg.det(np.einsum("eai,qaj->eqij", X[T], dNm))
        Me = np.einsum("e,q,qa,qb,eq->eab", rho, wm, Nm, Nm, detJm)
        r = np.repeat(T, 10, axis=1).ravel()
        c = np.tile(T, (1, 10)).ravel()
        Ms = sp.csr_matrix((Me.ravel(), (r, c)), shape=(self.n, self.n))
        self.node_mass = np.asarray(Ms.sum(1)).ravel()  # row sums (exact first moments)
        self.M = sp.kron(Ms, sp.identity(3), format="csr")

        dofs = (3 * T[:, :, None] + np.arange(3)).reshape(ne, 30)
        self.rows = np.repeat(dofs, 30, axis=1).ravel()
        self.cols = np.tile(dofs, (1, 30)).ravel()

        self.free = np.ones(self.ndof, bool)
        for a in fixed_nodes:
            self.free[3 * a : 3 * a + 3] = False

    def gravity(self, gvec):
        return self.M @ np.tile(gvec, self.n)

    def elastic(self, x, need_hess=True):
        """Psi(x), f_int = dPsi/dx, K_t = d2Psi/dx2 (sparse)."""
        xe = x.reshape(-1, 3)[self.T]
        F = np.einsum("eai,eqaJ->eqiJ", xe, self.gradN)
        Eg = 0.5 * (np.einsum("eqkI,eqkJ->eqIJ", F, F) - I3)
        trE = np.trace(Eg, axis1=2, axis2=3)
        mu, lam = self.mu[:, None], self.lam[:, None]
        psi = mu * (Eg * Eg).sum((2, 3)) + 0.5 * lam * trE**2
        S = 2 * mu[..., None, None] * Eg + (lam * trE)[..., None, None] * I3
        P = F @ S
        fe = np.einsum("eqiJ,eqaJ,eq->eai", P, self.gradN, self.dV)
        f = np.zeros((self.n, 3))
        np.add.at(f, self.T, fe)
        Psi = float((psi * self.dV).sum())
        if not need_hess:
            return Psi, f.ravel(), None
        A = np.einsum("ik,eqJL->eqiJkL", I3, S)
        A += self.lam[:, None, None, None, None, None] * np.einsum("eqiJ,eqkL->eqiJkL", F, F)
        FFt = np.einsum("eqiM,eqkM->eqik", F, F)
        A += self.mu[:, None, None, None, None, None] * (
            np.einsum("eqik,JL->eqiJkL", FFt, I3) + np.einsum("eqiL,eqkJ->eqiJkL", F, F)
        )
        Ke = np.einsum("eqaJ,eqiJkL,eqbL,eq->eaibk", self.gradN, A, self.gradN, self.dV, optimize=True)
        Ke = Ke.reshape(len(self.T), 30, 30)
        if self.project_elastic:
            Ke = psd_project(0.5 * (Ke + Ke.transpose(0, 2, 1)))
        K = sp.csr_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof))
        return Psi, f.ravel(), K


# ---------------------------------------------------------------------------
# Barrier: squared-distance clamped log, "physical" scaling (matches ipctk
# BarrierPotential(dhat, kappa, use_physical_barrier=True) with area weighting).
#   b(d) = -(d^2 - dhat^2)^2 ln(d^2/dhat^2) / dhat^3   for d < dhat
# B(x) = sum_v A_v b(d_v)  ->  kappa has units of pressure; near contact the
# pressure is p ~ 2 kappa dhat / d, so the gap at pressure p is ~ 2 kappa dhat / p.
# ---------------------------------------------------------------------------
def barrier(d, dhat):
    s, sh = d * d, dhat * dhat
    act = d < dhat
    ls = np.log(np.where(act, s / sh, 1.0))
    g0 = np.where(act, -((s - sh) ** 2) * ls, 0.0)
    g1 = np.where(act, -2 * (s - sh) * ls - (s - sh) ** 2 / s, 0.0)
    g2 = np.where(act, -2 * ls - 4 * (s - sh) / s + (s - sh) ** 2 / s**2, 0.0)
    k = 1.0 / dhat**3
    return k * g0, k * 2 * d * g1, k * (4 * s * g2 + 2 * g1)


def f1_friction(y, eps):
    """Smoothed Coulomb mollifier f1 (IPC 2020) and the PSD Hessian pieces."""
    inside = y < eps
    f1 = np.where(inside, -(y * y) / eps**2 + 2 * y / eps, 1.0)
    f1_over_y = np.where(inside, 2 / eps - y / eps**2, 1.0 / np.maximum(y, 1e-300))
    F0 = np.where(inside, -(y**3) / (3 * eps**2) + y * y / eps + eps / 3, y)
    return F0, f1, f1_over_y


def tangent_basis(n):
    a = np.where(np.abs(n[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return np.stack([t1, np.cross(n, t1)], axis=2)  # (k,3,2)


# ---------------------------------------------------------------------------
# Contact: analytic rigid box
# ---------------------------------------------------------------------------
class RigidBox:
    def __init__(self, lo, hi):
        self.c = 0.5 * (np.asarray(lo) + np.asarray(hi))
        self.hw = 0.5 * (np.asarray(hi) - np.asarray(lo))

    def distance(self, p):
        """Unsigned exterior distance d, unit normal, Hessian of d (convex in p)."""
        r = p - self.c
        q = np.abs(r) - self.hw
        qp = np.maximum(q, 0.0)
        d = np.linalg.norm(qp, axis=1)
        ds = np.maximum(d, 1e-300)
        n = np.sign(r) * qp / ds[:, None]
        mask = (q > 0).astype(float)
        Hd = (mask[:, :, None] * I3 - n[:, :, None] * n[:, None, :]) / ds[:, None, None]
        return d, n, Hd

    def signed_distance(self, p):
        r = p - self.c
        q = np.abs(r) - self.hw
        d, n, _ = self.distance(p)
        inside = (q <= 0).all(1)
        ax = q.argmax(1)
        n_in = np.zeros_like(p)
        n_in[np.arange(len(p)), ax] = np.sign(r[np.arange(len(p)), ax])
        return np.where(inside, q.max(1), d), np.where(inside[:, None], n_in, n)


class RigidBoxIPC:
    """IPC against an analytic obstacle: barrier, friction, CCD written out explicitly.
    These are the per-vertex kernels a CUDA version needs (point vs SDF)."""

    implicit = True

    def __init__(self, box, surf, area, dhat, kappa, mu=0.0, eps_v=1e-3):
        self.box, self.surf, self.A = box, surf, area[surf]
        self.dhat, self.kappa, self.mu, self.eps_v = dhat, kappa, mu, eps_v
        self.fr = None

    # lagged friction data: normal force magnitude and tangent basis at x_n
    def begin_step(self, x_n, v_n):
        if self.mu <= 0:
            return
        p = x_n.reshape(-1, 3)[self.surf]
        d, n, _ = self.box.distance(p)
        act = np.nonzero(d < self.dhat)[0]
        _, b1, _ = barrier(d[act], self.dhat)
        lam = -self.kappa * self.A[act] * b1  # normal force magnitude
        self.fr = (act, lam, tangent_basis(n[act]))

    def phi(self, x, v, h):
        p = x.reshape(-1, 3)[self.surf]
        d, _, _ = self.box.distance(p)
        if (d <= 0).any():
            return np.inf
        b0, _, _ = barrier(d, self.dhat)
        val = self.kappa * float((self.A * b0).sum()) / h
        if self.fr is not None and len(self.fr[0]):
            act, lam, Tb = self.fr
            u = np.einsum("kij,ki->kj", Tb, v.reshape(-1, 3)[self.surf[act]])
            F0, _, _ = f1_friction(np.linalg.norm(u, axis=1), self.eps_v)
            val += self.mu * float((lam * F0).sum())
        return val

    def grad_hess(self, x, v, h, ndof):
        p = x.reshape(-1, 3)[self.surf]
        d, n, Hd = self.box.distance(p)
        act = np.nonzero(d < self.dhat)[0]
        _, b1, b2 = barrier(d[act], self.dhat)
        kA = self.kappa * self.A[act]
        g = (kA * b1)[:, None] * n[act]
        Hb = kA[:, None, None] * (b2[:, None, None] * n[act, :, None] * n[act, None, :] + b1[:, None, None] * Hd[act])
        Hb = h * psd_project(Hb)
        nodes = self.surf[act]
        if self.fr is not None and len(self.fr[0]):
            fa, lam, Tb = self.fr
            u = np.einsum("kij,ki->kj", Tb, v.reshape(-1, 3)[self.surf[fa]])
            y = np.linalg.norm(u, axis=1)
            _, f1, f1y = f1_friction(y, self.eps_v)
            gf = self.mu * (lam * f1y)[:, None] * np.einsum("kij,kj->ki", Tb, u)
            uh = u / np.maximum(y, 1e-300)[:, None]
            inside = (y < self.eps_v)[:, None, None]
            corr = np.where(inside, -(y / self.eps_v**2)[:, None, None], -f1y[:, None, None]) * uh[:, :, None] * uh[:, None, :]
            H2 = f1y[:, None, None] * np.eye(2) + corr
            Hf = self.mu * lam[:, None, None] * np.einsum("kia,kab,kjb->kij", Tb, H2, Tb)
            g = np.concatenate([g, gf])
            Hb = np.concatenate([Hb, Hf])
            nodes = np.concatenate([nodes, self.surf[fa]])
        G = np.zeros(ndof)
        np.add.at(G.reshape(-1, 3), nodes, g)
        dof = 3 * nodes[:, None] + np.arange(3)
        r = np.repeat(dof, 3, axis=1).ravel()
        c = np.tile(dof, (1, 3)).ravel()
        H = sp.csr_matrix((Hb.ravel(), (r, c)), shape=(ndof, ndof))
        return G, H, len(act)

    def max_step(self, x, dx, s=0.1):
        """Largest alpha in [0,1] keeping every vertex at >= s * its current distance.

        d(p + a dp) is convex in a (distance to a convex set along a line), so a
        golden-section search finds the closest approach and a bisection on the
        decreasing branch finds the first crossing. Exact CCD for point vs convex SDF."""
        p = x.reshape(-1, 3)[self.surf]
        dp = dx.reshape(-1, 3)[self.surf]
        d0, _, _ = self.box.distance(p)
        target = s * d0
        dist = lambda a: self.box.distance(p + a[:, None] * dp)[0]
        lo, hi = np.zeros(len(p)), np.ones(len(p))
        gr = (np.sqrt(5) - 1) / 2
        for _ in range(60):
            m1, m2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
            left = dist(m1) < dist(m2)
            hi = np.where(left, m2, hi)
            lo = np.where(left, lo, m1)
        a_min = 0.5 * (lo + hi)
        hit = dist(a_min) < target
        if not hit.any():
            return 1.0
        lo, hi = np.zeros(hit.sum()), a_min[hit]
        pp, dd, tt = p[hit], dp[hit], target[hit]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            ok = self.box.distance(pp + mid[:, None] * dd)[0] >= tt
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return float(lo.min())

    def min_distance(self, x):
        return float(self.box.distance(x.reshape(-1, 3)[self.surf])[0].min())


class RigidBoxPenalty:
    """Explicit lagged penalty: pressure k*delta from start-of-step positions, frozen
    in f_ext for the whole Newton solve. Same coupling as the DEME/hydro path today."""

    implicit = False

    def __init__(self, box, surf, area, k_pen, mu=0.0, eps_v=1e-3):
        self.box, self.surf, self.A = box, surf, area[surf]
        self.k, self.mu, self.eps_v = k_pen, mu, eps_v

    def explicit_force(self, x_n, v_n, ndof):
        p = x_n.reshape(-1, 3)[self.surf]
        sd, n = self.box.signed_distance(p)
        fn = self.k * self.A * np.maximum(-sd, 0.0)
        f = fn[:, None] * n
        if self.mu > 0:
            vv = v_n.reshape(-1, 3)[self.surf]
            vt = vv - np.einsum("ki,ki->k", vv, n)[:, None] * n
            s = np.linalg.norm(vt, axis=1)
            f -= (self.mu * fn / np.maximum(s, self.eps_v))[:, None] * vt
        F = np.zeros(ndof)
        np.add.at(F.reshape(-1, 3), self.surf, f)
        return F, int((sd < 0).sum())

    def min_distance(self, x):
        return float(self.box.signed_distance(x.reshape(-1, 3)[self.surf])[0].min())


# ---------------------------------------------------------------------------
# Contact: deformable mesh vs deformable mesh through ipc-toolkit
# ---------------------------------------------------------------------------
class MeshIPC:
    """Convergent IPC (area weighting + improved max approximation) via ipctk."""

    implicit = True

    def __init__(self, X, tris, dhat, kappa, mu=0.0, eps_v=1e-3):
        import ipctk

        if os.environ.get("IPCTK_THREADS"):
            ipctk.set_num_threads(int(os.environ["IPCTK_THREADS"]))
        self.ipctk = ipctk
        self.cm = ipctk.CollisionMesh.build_from_full_mesh(X, ipctk.edges(tris), tris)
        self.dhat, self.kappa, self.mu = dhat, kappa, mu
        self.B = ipctk.BarrierPotential(dhat, kappa, True)
        self.fp = ipctk.FrictionPotential(eps_v)
        self.tc = None

    def _collisions(self, V):
        c = self.ipctk.NormalCollisions()
        c.use_area_weighting = True
        c.collision_set_type = self.ipctk.NormalCollisions.CollisionSetType.IMPROVED_MAX_APPROX
        c.build(self.cm, V, self.dhat)
        return c

    def _V(self, x):
        return self.cm.vertices(np.asfortranarray(x.reshape(-1, 3)))

    def begin_step(self, x_n, v_n):
        if self.mu <= 0:
            return
        V = self._V(x_n)
        self.tc = self.ipctk.TangentialCollisions()
        self.tc.build(self.cm, V, self._collisions(V), self.B, self.mu)

    def _vel(self, v):
        return self.cm.map_displacements(np.asfortranarray(v.reshape(-1, 3)))

    def phi(self, x, v, h):
        V = self._V(x)
        col = self._collisions(V)
        if len(col.fv_collisions) + len(col.ee_collisions) + len(col.ev_collisions) + len(col.vv_collisions):
            if col.compute_minimum_distance(self.cm, V) <= 0:
                return np.inf
        val = self.B(col, self.cm, V) / h
        if self.tc is not None:
            val += self.fp(self.tc, self.cm, self._vel(v))
        return val

    def grad_hess(self, x, v, h, ndof):
        V = self._V(x)
        col = self._collisions(V)
        PSD = self.ipctk.PSDProjectionMethod.CLAMP
        G = self.cm.to_full_dof(self.B.gradient(col, self.cm, V))
        H = h * self.cm.to_full_dof(self.B.hessian(col, self.cm, V, PSD))
        if self.tc is not None:
            Vel = self._vel(v)
            G = G + self.cm.to_full_dof(self.fp.gradient(self.tc, self.cm, Vel))
            H = H + self.cm.to_full_dof(self.fp.hessian(self.tc, self.cm, Vel, PSD))
        n_act = len(col.fv_collisions) + len(col.ee_collisions) + len(col.ev_collisions) + len(col.vv_collisions)
        return np.asarray(G).ravel(), sp.csr_matrix(H), n_act

    def max_step(self, x, dx, s=0.1):
        # Same rule as the analytic path: never close more than (1 - s) of the current
        # gap in one step (C-IPC's minimum-separation CCD). Without a minimum separation
        # a step can land exactly on d = 0 and the barrier becomes meaningless.
        V0 = self._V(x)
        V1 = self._V(x + dx)
        md = s * self.min_distance(x, net=self.dhat)
        return float(self.ipctk.compute_collision_free_stepsize(self.cm, V0, V1, min_distance=md))

    def min_distance(self, x, net=0.05):
        # ipctk works with squared distances internally; return a length.
        V = self._V(x)
        col = self.ipctk.NormalCollisions()
        col.build(self.cm, V, net)
        n = len(col.fv_collisions) + len(col.ee_collisions) + len(col.ev_collisions) + len(col.vv_collisions)
        return float(np.sqrt(col.compute_minimum_distance(self.cm, V))) if n else net


# ---------------------------------------------------------------------------
# Time step: SyncedNewton's velocity-form Newton + IPC line search
# ---------------------------------------------------------------------------
@dataclass
class StepStats:
    newton: int = 0
    ls_backtracks: int = 0
    ccd_min_alpha: float = 1.0
    active: int = 0
    converged: bool = True
    kappa: float = 0.0


def newton_step(model, contact, x_n, v_n, h, f_ext, opt, stats, x_start=None):
    """Minimize Phi(v) with x = x_n + h v.  BE: (x_n, v_n, h).  BDF2 reuses this with
    x_n <- (4x_n - x_{n-1})/3, v_n <- (4v_n - v_{n-1})/3, h <- 2h/3."""
    free = model.free
    M = model.M
    v = v_n.copy()
    v[~free] = 0.0
    if contact is not None and contact.implicit:
        # IPC needs a feasible first iterate. Start at x_start (the last accepted,
        # intersection-free positions) and move toward the predictor as far as CCD allows.
        x_start = x_n if x_start is None else x_start
        v0 = (x_start - x_n) / h
        v = v0 + contact.max_step(x_start, x_n + h * v - x_start) * (v - v0)

    def phi(v):
        x = x_n + h * v
        Psi, _, _ = model.elastic(x, need_hess=False)
        dv = v - v_n
        val = 0.5 / h * dv @ (M @ dv) + (Psi - f_ext @ x) / h
        if contact is not None and contact.implicit:
            val += contact.phi(x, v, h)
        return val

    prev_dmin = np.inf
    for it in range(opt.max_newton):
        x = x_n + h * v
        _, f_int, K = model.elastic(x)
        g = M @ (v - v_n) / h + f_int - f_ext
        H = M / h + h * K
        if contact is not None and contact.implicit:
            gc, Hc, n_act = contact.grad_hess(x, v, h, model.ndof)
            g, H = g + gc, H + Hc
            stats.active = max(stats.active, n_act)
        dv = np.zeros_like(v)
        if free.all():
            dv = spla.spsolve(H.tocsc(), -g)
        else:
            idx = np.nonzero(free)[0]
            dv[idx] = spla.spsolve(H.tocsr()[idx][:, idx].tocsc(), -g[idx])
        stats.newton = it + 1
        if np.abs(dv).max() < opt.tol_v:
            return v
        alpha = 1.0
        if contact is not None and contact.implicit:
            alpha = min(1.0, contact.max_step(x, h * dv))
            stats.ccd_min_alpha = min(stats.ccd_min_alpha, alpha)
        if opt.line_search:
            phi0, slope = phi(v), g @ dv
            while phi(v + alpha * dv) > phi0 + 1e-4 * alpha * slope and alpha > 1e-10:
                alpha *= 0.5
                stats.ls_backtracks += 1
        v = v + alpha * dv
        if opt.verbose_newton:
            print(f"    it={it} |g|={np.linalg.norm(g[free]):.3e} |dv|inf={np.abs(dv).max():.3e} alpha={alpha:.3e}")
        if contact is not None and contact.implicit and opt.adaptive_kappa:
            dmin = contact.min_distance(x_n + h * v)
            if dmin < opt.kappa_eps * contact.dhat and dmin < prev_dmin:
                contact.kappa *= 2.0
                if hasattr(contact, "B"):
                    contact.B = contact.ipctk.BarrierPotential(contact.dhat, contact.kappa, True)
            prev_dmin = dmin
    stats.converged = False
    return v


# ---------------------------------------------------------------------------
# Scenes and driver
# ---------------------------------------------------------------------------
@dataclass
class Options:
    scene: str = "rigid"
    contact: str = "ipc"
    dt: float = 1e-3
    t_end: float = 0.015
    v0: float = 1.0
    vx0: float = 0.0
    gap0: float = 2e-3
    gravity: float = 0.0
    E: float = 1e7
    nu: float = 0.3
    rho: float = 70.7355  # 1 kg sphere, as in lib_bin/ptest
    E_box: float = 1e7
    mesh: str = "data/meshes/ptest/sphere/sphere_r0p15_low.1"
    # Barrier range and stiffness are fixed per scene for now (decision 2026-10-12).
    # kappa is a pressure: at contact pressure p the gap settles near 2*kappa*dhat/p.
    # These defaults give ~25 um at the 1 m/s impact peak (~1 MPa) but ~0.66 mm when
    # the sphere rests on the box (~30 kPa). Lower kappa or dhat for a smaller resting gap.
    dhat: float = 1e-3
    kappa: float = 1e4
    k_pen: float = 1e9
    mu: float = 0.0
    eps_v: float = 1e-3
    lag_iters: int = 1
    max_newton: int = 60
    tol_v: float = 1e-6
    line_search: bool = True
    adaptive_kappa: bool = True
    kappa_eps: float = 1e-3
    box_res: tuple = (10, 10, 2)
    xy_offset: tuple = (0.0113, 0.0071)
    integrator: str = "bdf2"  # default for contact runs; "be" = SyncedNewton's backward Euler
    out: str = ""
    quiet: bool = False
    verbose_newton: bool = False
    record: str = ""  # .npz path: node positions at every step


def build_scene(opt):
    Xs, Ts = read_tetgen_t10(os.path.join(REPO, opt.mesh))
    R = np.linalg.norm(Xs, axis=1).max()
    box_lo, box_hi = np.array([-0.3, -0.3, -0.1]), np.array([0.3, 0.3, 0.0])
    # Small lateral offset so the sphere's lowest node is not exactly above a box node
    # or edge (a measure-zero configuration where the distance Hessian has a kink).
    Xs = Xs + np.array([opt.xy_offset[0], opt.xy_offset[1], box_hi[2] + R + opt.gap0])
    surf_tris_s = surface_triangles_4way(Ts)
    if opt.scene == "rigid":
        model = T10Model(Xs, Ts, opt.E, opt.nu, opt.rho)
        surf = np.unique(surf_tris_s)
        area = vertex_areas(Xs, surf_tris_s, len(Xs))
        box = RigidBox(box_lo, box_hi)
        if opt.contact == "ipc":
            contact = RigidBoxIPC(box, surf, area, opt.dhat, opt.kappa, opt.mu, opt.eps_v)
        else:
            contact = RigidBoxPenalty(box, surf, area, opt.k_pen, opt.mu, opt.eps_v)
        sphere_nodes = np.arange(len(Xs))
        return model, contact, sphere_nodes, Xs
    # fem: T10 box (bottom fixed) + T10 sphere, one global system
    Xb, Tb = box_t10(box_lo, box_hi, opt.box_res)
    nb = len(Xb)
    X = np.vstack([Xb, Xs])
    T = np.vstack([Tb, Ts + nb])
    E = np.r_[np.full(len(Tb), opt.E_box), np.full(len(Ts), opt.E)]
    fixed = np.nonzero(np.isclose(Xb[:, 2], box_lo[2]))[0]
    model = T10Model(X, T, E, opt.nu, opt.rho, fixed_nodes=fixed)
    tris = np.vstack([surface_triangles_4way(Tb), surf_tris_s + nb]).astype(np.int32)
    contact = MeshIPC(X, tris, opt.dhat, opt.kappa, opt.mu, opt.eps_v)
    return model, contact, nb + np.arange(len(Xs)), X


def run(opt: Options):
    model, contact, sph, X0 = build_scene(opt)
    h = opt.dt
    n_steps = int(round(opt.t_end / h))
    x = X0.ravel().copy()
    v = np.zeros(model.ndof)
    v.reshape(-1, 3)[sph] = [opt.vx0, 0.0, -opt.v0]
    f_g = model.gravity(np.array([0.0, 0.0, -opt.gravity]))
    m_s = model.node_mass[sph]
    m_tot = m_s.sum()
    com = lambda q: (m_s[:, None] * q.reshape(-1, 3)[sph]).sum(0) / m_tot
    hist = []
    frames = [X0.reshape(-1, 3).copy()]
    t0 = time.time()
    for k in range(n_steps):
        st = StepStats()
        f_ext = f_g.copy()
        n_pen = 0
        if contact is not None and not contact.implicit:
            fc, n_pen = contact.explicit_force(x, v, model.ndof)
            f_ext = f_ext + fc
        if opt.integrator == "bdf2" and k > 0:
            xb, vb, he = (4 * x - x_old) / 3, (4 * v - v_old) / 3, 2 * h / 3
        else:
            xb, vb, he = x.copy(), v.copy(), h
        x_old, v_old = x.copy(), v.copy()
        if contact is not None and contact.implicit:
            # Friction is lagged: normal forces and tangent bases come from x_n.
            # Extra lag iterations re-evaluate them at the latest solution.
            contact.begin_step(x_old, v_old)
            v_new = newton_step(model, contact, xb, vb, he, f_ext, opt, st, x_start=x_old)
            for _ in range(opt.lag_iters - 1 if opt.mu > 0 else 0):
                contact.begin_step(xb + he * v_new, v_old)
                v_new = newton_step(model, contact, xb, vb, he, f_ext, opt, st, x_start=x_old)
        else:
            v_new = newton_step(model, contact, xb, vb, he, f_ext, opt, st)
        v = v_new
        x = xb + he * v
        Psi, _, _ = model.elastic(x, need_hess=False)
        KE = 0.5 * v @ (model.M @ v)
        c, cv = com(x), com(v)
        dmin = contact.min_distance(x) if contact is not None else np.inf
        hist.append(
            dict(
                t=(k + 1) * h, com_x=c[0], com_z=c[2], com_vx=cv[0], com_vz=cv[2], dmin=dmin,
                KE=KE, Psi=Psi, PE_g=-(f_g @ x), newton=st.newton, ls=st.ls_backtracks,
                ccd=st.ccd_min_alpha, active=st.active or n_pen, conv=st.converged,
                kappa=getattr(contact, "kappa", 0.0),
            )
        )
        frames.append(x.reshape(-1, 3).copy())
        if not np.isfinite(x).all() or np.abs(v).max() > 1e3:
            hist[-1]["blowup"] = True
            break
        if not opt.quiet and (k % max(1, n_steps // 20) == 0 or k == n_steps - 1):
            r = hist[-1]
            print(f"  t={r['t']:.4f} z={r['com_z']:.5f} vz={r['com_vz']:+.4f} dmin={r['dmin']:+.3e} "
                  f"newton={r['newton']} ccd={r['ccd']:.3f} act={r['active']} KE={KE:.4f} Psi={Psi:.4f}")
    wall = time.time() - t0
    if opt.record:
        # Node positions at every step, for replaying the run (viewer / post-processing).
        np.savez_compressed(
            opt.record, X0=X0.reshape(-1, 3), T=model.T, frames=np.array(frames), sphere_nodes=sph,
            t=np.r_[0.0, [r["t"] for r in hist]], dmin=np.r_[np.nan, [r["dmin"] for r in hist]],
            com_vz=np.r_[-opt.v0, [r["com_vz"] for r in hist]], scene=opt.scene, contact=opt.contact,
            integrator=opt.integrator, dt=opt.dt)
    return summarize(opt, hist, wall), hist


def summarize(opt, hist, wall):
    vz_in = -opt.v0
    vz_out = hist[-1]["com_vz"]
    dmin = min(r["dmin"] for r in hist)
    E0 = 0.5 * 1.0 * (opt.v0**2 + opt.vx0**2)
    E_end = hist[-1]["KE"] + hist[-1]["Psi"]
    return dict(
        scene=opt.scene, contact=opt.contact, dt=opt.dt, steps=len(hist),
        blowup=bool(hist[-1].get("blowup", False)),
        cor=(vz_out / -vz_in) if opt.gravity == 0 else None,
        vx_out=hist[-1]["com_vx"],
        min_gap_or_sd=dmin, max_penetration=max(0.0, -dmin),
        energy_ratio=E_end / E0 if E0 > 0 else None,
        newton_total=int(sum(r["newton"] for r in hist)),
        newton_max=int(max(r["newton"] for r in hist)),
        unconverged=int(sum(not r["conv"] for r in hist)),
        min_ccd_alpha=float(min(r["ccd"] for r in hist)),
        kappa_end=hist[-1]["kappa"], wall_s=round(wall, 1),
        in_contact_at_end=bool(hist[-1]["active"] > 0 or hist[-1]["dmin"] < opt.dhat),
    )


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Options()
    for f, val in asdict(d).items():
        if isinstance(val, bool):
            p.add_argument("--" + f.replace("_", "-"), type=lambda s: s.lower() in ("1", "true", "yes"), default=val)
        elif isinstance(val, tuple):
            cast = type(val[0])
            p.add_argument("--" + f.replace("_", "-"), type=lambda s, c=cast: tuple(c(t) for t in s.split(",")), default=val)
        else:
            p.add_argument("--" + f.replace("_", "-"), type=type(val), default=val)
    p.add_argument("--suite", action="store_true", help="run every case used in the write-up")
    a = vars(p.parse_args())
    suite = a.pop("suite")
    return Options(**a), suite


def main():
    opt, suite = parse()
    if not suite:
        s, hist = run(opt)
        print(json.dumps(s, indent=1))
        if opt.out:
            os.makedirs(opt.out, exist_ok=True)
            json.dump(dict(summary=s, hist=hist), open(os.path.join(opt.out, "run.json"), "w"))
        return
    from suite import run_suite  # noqa: E402  (kept separate so the solver file stays readable)

    run_suite(run, Options, opt.out or os.path.join(os.path.dirname(__file__), "results"))


if __name__ == "__main__":
    main()
