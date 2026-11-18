#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "ipctk==1.6.0"]
# ///
"""
Verification of the IPC prototype: does the code implement the equations it claims, and do
its discrete solutions obey the conservation and contact laws they should? No comparison
with analytic benchmarks or experiments here (that is hertz.py / suite.py).

    uv run test-scripts/ipc_proto/verify.py

Sections
    A  T10 element: shape functions, quadrature, mass, SVK energy/forces/stiffness
    B  contact surface: 4-way split, vertex areas
    C  barrier b(d)
    D  rigid-box contact: distance, barrier gradient/Hessian, CCD, friction
    E  mesh-mesh contact (ipctk): gradient/Hessian, invariances, CCD, barrier scale
    F  solver: assembled residual/Hessian vs Phi, Newton convergence
    G  time stepping: free fall, momentum, energy, intersection-free iterates
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
import time
import types
from collections import Counter
from dataclasses import replace
from math import factorial

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sphere_on_box as sob  # noqa: E402

RNG = np.random.default_rng(7)
ROWS = []


def check(sec, name, value, ok, note=""):
    tag = {True: "PASS", False: "FAIL", None: "NOTE"}[ok]
    ROWS.append((sec, name, value, tag, note))
    print(f"[{tag}] {sec} {name}: {value}" + (f"  ({note})" if note else ""), flush=True)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def fd(f, x, w, eps):
    return (f(x + eps * w) - f(x - eps * w)) / (2 * eps)


def unit(w):
    return w / np.abs(w).max()


def rotation():
    Q, R = np.linalg.qr(RNG.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))
    return Q if np.linalg.det(Q) > 0 else -Q


OPT = sob.Options(quiet=True)
MESH = os.path.join(sob.REPO, OPT.mesh)


def sphere():
    return sob.read_tetgen_t10(MESH)


# ---------------------------------------------------------------------------
# A. T10 element
# ---------------------------------------------------------------------------
def section_a():
    S = "A"
    xi = RNG.dirichlet(np.ones(4), 64)[:, 1:]
    N, dN = sob.t10_shape(xi), sob.t10_shape_grads(xi)
    C = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    nodes = np.vstack([C, [(C[i] + C[j]) / 2 for i, j in sob.EDGES]])
    e = 1e-6
    dN_fd = np.stack([(sob.t10_shape(xi + e * I) - sob.t10_shape(xi - e * I)) / (2 * e) for I in np.eye(3)], axis=2)
    err = max(np.abs(sob.t10_shape(nodes) - np.eye(10)).max(), np.abs(N.sum(1) - 1).max(), np.abs(dN.sum(1)).max())
    check(S, "1 shape fns: N_a(node_b)=delta_ab, sum N=1, sum dN=0", f"{err:.1e}", err < 1e-13)
    check(S, "2 shape fn gradients = FD of N", f"{np.abs(dN_fd - dN).max():.1e}", np.abs(dN_fd - dN).max() < 1e-8)

    exact = lambda a, b, c: factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)

    def qerr(rule, deg):
        p, w = rule
        return max(abs((w * p[:, 0] ** a * p[:, 1] ** b * p[:, 2] ** c).sum() / exact(a, b, c) - 1)
                   for a in range(deg + 1) for b in range(deg + 1 - a) for c in range(deg + 1 - a - b))

    e4, ec = qerr(sob.tet_rule_4pt(), 2), qerr(sob.tet_rule_collapsed(4), 5)
    check(S, "3 quadrature: 4-pt exact to degree 2 (stiffness), collapsed to degree 5 (mass)", f"{e4:.1e}, {ec:.1e}",
          max(e4, ec) < 1e-12)

    Xs, Ts = sphere()
    E, nu, rho = OPT.E, OPT.nu, OPT.rho
    m = sob.T10Model(Xs, Ts, E, nu, rho)
    mnp = sob.T10Model(Xs, Ts, E, nu, rho, project_elastic=False)
    Ms = m.M[0::3, 0::3].toarray()
    ev = np.linalg.eigvalsh(Ms)
    curved = np.abs(Xs[Ts[:, 4:]] - 0.5 * (Xs[Ts[:, [i for i, _ in sob.EDGES]]] + Xs[Ts[:, [j for _, j in sob.EDGES]]])).max()
    xm, wm = sob.tet_rule_collapsed(7)
    Nm = sob.t10_shape(xm)
    detJ = np.linalg.det(np.einsum("eai,qaj->eqij", Xs[Ts], sob.t10_shape_grads(xm)))
    Me_hi = np.einsum("q,qa,qb,eq->eab", wm, Nm, Nm, detJ) * rho
    xm4, wm4 = sob.tet_rule_collapsed(4)
    detJ4 = np.linalg.det(np.einsum("eai,qaj->eqij", Xs[Ts], sob.t10_shape_grads(xm4)))
    Me_4 = np.einsum("q,qa,qb,eq->eab", wm4, sob.t10_shape(xm4), sob.t10_shape(xm4), detJ4) * rho
    check(S, "4 mass: consistent M positive definite, rule exact (vs 7-pt collapsed)",
          f"min/max eig {ev[0] / ev[-1]:.2e}, rule diff {np.abs(Me_4 - Me_hi).max() / np.abs(Me_hi).max():.1e}",
          ev[0] > 0 and np.abs(Me_4 - Me_hi).max() < 1e-12 * np.abs(Me_hi).max(),
          f"m = {m.node_mass.sum():.5f} kg vs 1 kg sphere; midside nodes off straight edges by {curved:.1e} m")
    pk = np.array([[.25, .25, .25], [1 / 6, 1 / 6, 1 / 6], [.5, 1 / 6, 1 / 6], [1 / 6, .5, 1 / 6], [1 / 6, 1 / 6, .5]])
    Nk = sob.t10_shape(pk)
    Mk = np.einsum("q,qa,qb->ab", np.array([-2 / 15, 3 / 40, 3 / 40, 3 / 40, 3 / 40]), Nk, Nk)
    check(S, "5 contrast: repo Keast-5 rule on the reference element mass", f"min eig {np.linalg.eigvalsh(Mk)[0]:+.2e}", None,
          "negative: why the prototype does not use it")

    # patch test: uniform F -> Psi = psi(F) * V exactly and zero force on interior nodes
    F0 = np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.015], [0.005, 0.0, 1.01]])
    xp = (Xs @ F0.T).ravel()
    Psi_p, f_p, _ = mnp.elastic(xp, need_hess=False)
    Eg = 0.5 * (F0.T @ F0 - np.eye(3))
    psi = mnp.mu[0] * (Eg * Eg).sum() + 0.5 * mnp.lam[0] * np.trace(Eg) ** 2
    surf = np.unique(sob.surface_triangles_4way(Ts))
    inner = np.setdiff1d(np.arange(len(Xs)), surf)
    fr = f_p.reshape(-1, 3)
    check(S, "6 patch test: Psi = psi(F)V, interior nodal force = 0 under uniform F",
          f"Psi rel err {abs(Psi_p / (psi * mnp.dV.sum()) - 1):.1e}, interior |f|/max|f| {np.abs(fr[inner]).max() / np.abs(fr).max():.1e}",
          abs(Psi_p / (psi * mnp.dV.sum()) - 1) < 1e-12 and np.abs(fr[inner]).max() < 1e-10 * np.abs(fr).max())

    # frame indifference
    Q, c = rotation(), RNG.normal(size=3)
    Psi_r, f_r, _ = mnp.elastic((Xs @ Q.T + c).ravel(), need_hess=False)
    xd = (0.97 * Xs + 2e-4 * RNG.normal(size=Xs.shape)).ravel()
    Psi_d, f_d, K_d = mnp.elastic(xd)
    Psi_dr, f_dr, _ = mnp.elastic((xd.reshape(-1, 3) @ Q.T + c).ravel(), need_hess=False)
    check(S, "7 frame indifference: rigid motion -> Psi=0, f=0; Psi(Qx+c)=Psi(x), f(Qx+c)=Qf(x)",
          f"{abs(Psi_r) / Psi_p:.1e}, {np.abs(f_r).max() / np.abs(f_p).max():.1e}; {abs(Psi_dr / Psi_d - 1):.1e}, "
          f"{rel(f_dr.reshape(-1, 3), f_d.reshape(-1, 3) @ Q.T):.1e}",
          max(abs(Psi_r) / Psi_p, np.abs(f_r).max() / np.abs(f_p).max(), abs(Psi_dr / Psi_d - 1),
              rel(f_dr.reshape(-1, 3), f_d.reshape(-1, 3) @ Q.T)) < 1e-9)
    fd_ = f_d.reshape(-1, 3)
    xr = xd.reshape(-1, 3)
    sc = np.abs(fd_).sum()
    check(S, "8 internal forces: sum f = 0, sum x cross f = 0 (momentum)",
          f"{np.abs(fd_.sum(0)).max() / sc:.1e}, {np.abs(np.cross(xr, fd_).sum(0)).max() / (sc * 0.15):.1e}",
          np.abs(fd_.sum(0)).max() < 1e-12 * sc and np.abs(np.cross(xr, fd_).sum(0)).max() < 1e-12 * sc * 0.15)

    errs_f, errs_K = [], []
    for _ in range(3):
        w = unit(RNG.normal(size=xd.size))
        errs_f.append(abs(fd(lambda x: mnp.elastic(x, need_hess=False)[0], xd, w, 1e-6) / (f_d @ w) - 1))
        errs_K.append(rel(fd(lambda x: mnp.elastic(x, need_hess=False)[1], xd, w, 1e-7), K_d @ w))
    check(S, "9 f_int = grad Psi, K_t = grad f_int (FD, deformed state)", f"{max(errs_f):.1e}, {max(errs_K):.1e}",
          max(errs_f) < 1e-6 and max(errs_K) < 1e-6)

    _, _, K_p = m.elastic(xd)
    w_np = np.linalg.eigvalsh(K_d.toarray())
    w_p = np.linalg.eigvalsh(K_p.toarray())
    check(S, "10 per-element PSD projection: assembled K_t PSD", f"min eig/max: unprojected {w_np[0] / w_np[-1]:+.1e}, "
          f"projected {w_p[0] / w_p[-1]:+.1e}", w_p[0] > -1e-12 * w_p[-1],
          f"projection changes K by {rel(K_p.toarray(), K_d.toarray()):.1e} (rel) at 3% compression")
    return Xs, Ts


# ---------------------------------------------------------------------------
# B. Contact surface
# ---------------------------------------------------------------------------
def section_b(Xs, Ts):
    S = "B"
    tris = sob.surface_triangles_4way(Ts)
    dirs = Counter((int(a), int(b)) for t in tris for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    closed = all(v == 1 for v in dirs.values()) and all((b, a) in dirs for a, b in dirs)
    P = Xs[tris]
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6
    Vfe = sob.T10Model(Xs, Ts, OPT.E, OPT.nu, OPT.rho).dV.sum()
    check(S, "1 4-way split surface closed, consistently oriented outward", f"closed={closed}, enclosed V / FE V = {vol / Vfe:.4f}",
          closed and vol > 0)
    A = sob.vertex_areas(Xs, tris, len(Xs))
    tri_area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()
    R = np.linalg.norm(Xs, axis=1).max()
    check(S, "2 vertex areas partition the surface (A_v = 1/3 of incident triangles)",
          f"sum A_v / surface area - 1 = {A.sum() / tri_area - 1:.1e}", abs(A.sum() / tri_area - 1) < 1e-13,
          f"surface = {tri_area / (4 * np.pi * R**2):.4f} x 4 pi R^2")


# ---------------------------------------------------------------------------
# C. Barrier
# ---------------------------------------------------------------------------
def section_c():
    S = "C"
    dh = 1e-3
    bref = lambda d: -((d * d - dh * dh) ** 2) * np.log(d * d / (dh * dh)) / dh**3
    d = np.geomspace(1e-4, 0.999, 200) * dh
    b0, b1, b2 = sob.barrier(d, dh)
    e = 1e-6 * d
    e1 = np.abs((bref(d + e) - bref(d - e)) / (2 * e) / b1 - 1).max()
    e2 = np.abs((sob.barrier(d + e, dh)[1] - sob.barrier(d - e, dh)[1]) / (2 * e) / b2 - 1).max()
    check(S, "1 b(d) = -(d^2-dhat^2)^2 ln(d^2/dhat^2)/dhat^3; b', b'' = FD", f"b {np.abs(b0 / bref(d) - 1).max():.1e}, "
          f"b' {e1:.1e}, b'' {e2:.1e}", max(np.abs(b0 / bref(d) - 1).max(), e1, e2) < 1e-6)
    at = np.abs(np.array(sob.barrier(np.array([dh]), dh)).ravel()).max()
    near = [np.abs(np.array(sob.barrier(np.array([dh * (1 - q)]), dh)).ravel()) / np.abs([b0, b1, b2]).max(1)
            for q in (1e-3, 1e-4)]
    order = np.log10(near[0] / near[1])  # 3, 2, 1 for a function that leaves zero like (dhat - d)^3
    check(S, "2 C2 switch-on: b = b' = b'' = 0 at d = dhat", f"at dhat {at:.0e}; decay order of b, b', b'' as d -> dhat: "
          + ", ".join(f"{q:.2f}" for q in order), at == 0 and np.allclose(order, [3, 2, 1], atol=0.05))
    check(S, "3 convex, decreasing, infinite wall", f"b>0 {bool((b0 > 0).all())}, b'<0 {bool((b1 < 0).all())}, "
          f"b''>0 {bool((b2 > 0).all())}, b(1e-12)/b(1e-4 dhat) = {sob.barrier(np.array([1e-12]), dh)[0][0] / b0[0]:.1f}",
          bool((b0 > 0).all() and (b1 < 0).all() and (b2 > 0).all()))
    r = [(-sob.barrier(np.array([q * dh]), dh)[1][0]) * q * dh / (2 * dh) for q in (1e-3, 2.5e-2, 1e-1, 0.5)]
    check(S, "4 pressure law p = -kappa b'(d) -> 2 kappa dhat / d as d << dhat",
          "p d/(2 kappa dhat) at d/dhat = 1e-3, 0.025, 0.1, 0.5: " + ", ".join(f"{v:.3f}" for v in r), abs(r[0] - 1) < 2e-2)
    import ipctk

    s, sh = d * d, dh * dh
    ib = np.array([ipctk.barrier(a, sh) for a in s]) / dh**3
    ib1 = np.array([ipctk.barrier_first_derivative(a, sh) for a in s]) * 2 * d / dh**3
    check(S, "5 same function as ipctk's physical barrier (applied to d^2)", f"b {np.abs(ib / b0 - 1).max():.1e}, b' {np.abs(ib1 / b1 - 1).max():.1e}",
          max(np.abs(ib / b0 - 1).max(), np.abs(ib1 / b1 - 1).max()) < 1e-12)


# ---------------------------------------------------------------------------
# D. Rigid box contact
# ---------------------------------------------------------------------------
def rigid_state(gap, dhat, mu, center_x=None):
    o = replace(OPT, scene="rigid", contact="ipc", gap0=gap, dhat=dhat, mu=mu)
    model, contact, sph, X0 = sob.build_scene(o)
    if center_x is not None:
        X0 = X0 + np.array([center_x - X0[:, 0].mean(), 0, 0])
    return o, model, contact, X0


def section_d():
    S = "D"
    box = sob.RigidBox(np.array([-0.3, -0.3, -0.1]), np.array([0.3, 0.3, 0.0]))
    p = np.array([0.3, 0.3, 0.0]) + RNG.uniform(-0.05, 0.05, (4000, 3))
    q = np.abs(p - box.c) - box.hw
    p = p[(box.distance(p)[0] > 1e-3) & (np.abs(q).min(1) > 1e-4)][:300]
    d, n, Hd = box.distance(p)
    e = 1e-7
    n_fd = np.stack([(box.distance(p + e * I)[0] - box.distance(p - e * I)[0]) / (2 * e) for I in np.eye(3)], 1)
    H_fd = np.stack([(box.distance(p + e * I)[1] - box.distance(p - e * I)[1]) / (2 * e) for I in np.eye(3)], 2)
    check(S, "1 box distance: n = grad d, Hd = grad n (face, edge, corner regions)",
          f"{np.abs(n_fd - n).max():.1e}, {np.abs(H_fd - Hd).max() / np.abs(Hd).max():.1e}",
          np.abs(n_fd - n).max() < 1e-7 and np.abs(H_fd - Hd).max() < 1e-6 * np.abs(Hd).max())

    h = 1e-3
    for label, cx in (("face", None), ("box edge", 0.302)):
        o, model, c, X0 = rigid_state(2e-4, 4e-3, 0.0, cx)
        x = X0.ravel() + 1e-5 * RNG.normal(size=X0.size)
        G, H, nact = c.grad_hess(x, np.zeros_like(x), h, model.ndof)
        assert nact > 0, label
        Bf = lambda y: h * c.phi(y, np.zeros_like(y), h)
        eg = max(abs(fd(Bf, x, w, 1e-9) / (G @ w) - 1) for w in [unit(RNG.normal(size=x.size)) for _ in range(3)])
        orig = sob.psd_project
        sob.psd_project = lambda Hb: Hb
        try:
            _, Hn, _ = c.grad_hess(x, np.zeros_like(x), h, model.ndof)
        finally:
            sob.psd_project = orig
        w = unit(RNG.normal(size=x.size))
        eh = rel(fd(lambda y: c.grad_hess(y, np.zeros_like(y), h, model.ndof)[0], x, w, 1e-9), Hn @ w / h)
        lam_min = min(np.linalg.eigvalsh(H.toarray()[np.ix_(idx, idx)])[0] for idx in
                      [np.r_[3 * a: 3 * a + 3] for a in c.surf[np.nonzero(c.box.distance(x.reshape(-1, 3)[c.surf])[0] < c.dhat)[0]]])
        check(S, f"2 barrier ({label}, {nact} active): kappa grad B = FD, kappa hess B = FD, projected blocks PSD",
              f"{eg:.1e}, {eh:.1e}, min eig {lam_min:+.1e}", eg < 1e-6 and eh < 1e-5 and lam_min > -1e-9)

        worst, maxim = np.inf, 0.0
        for _ in range(4):
            dx = h * (np.array([0.0, 0.0, -6.0]) + RNG.uniform(-0.5, 0.5, 3) * [1, 1, 0])
            dx = np.tile(dx, model.n) + h * 0.2 * RNG.normal(size=x.size)
            a = c.max_step(x, dx)
            pts, dps = x.reshape(-1, 3)[c.surf], dx.reshape(-1, 3)[c.surf]
            d0 = c.box.distance(pts)[0]
            ratio = np.array([(c.box.distance(pts + t * dps)[0] / d0).min() for t in np.linspace(0, a, 2001)])
            worst = min(worst, ratio.min())
            if a < 1:
                maxim = max(maxim, abs(ratio[-1] - 0.1))
        check(S, f"3 CCD ({label}): every vertex keeps >= 0.1 x its gap along [0, alpha]; alpha maximal",
              f"min d(t)/d0 = {worst:.6f}, |d(alpha)/d0 - 0.1| = {maxim:.1e}", worst >= 0.1 - 1e-9 and maxim < 1e-6)

    # friction
    o, model, c, X0 = rigid_state(5e-4, 4e-3, 0.3)
    x = X0.ravel()
    c.begin_step(x, np.zeros_like(x))
    act, lam, Tb = c.fr
    d, n, _ = c.box.distance(x.reshape(-1, 3)[c.surf])
    _, b1, _ = sob.barrier(d[act], c.dhat)
    check(S, f"4 friction normal force lambda = kappa A |b'(d)| at x_n ({len(act)} active)",
          f"{np.abs(lam / (c.kappa * c.A[act] * np.abs(b1)) - 1).max():.1e}", np.abs(lam / (c.kappa * c.A[act] * np.abs(b1)) - 1).max() < 1e-14)
    fr0 = c.fr
    ok_mag, ok_dir, ok_perp, ok_psd, diss, e_g, e_h = [], [], [], [], [], [], []
    for speed in (1e-5, 3e-4, 9e-4, 1e-3 * 1.5, 0.3, 2.0):
        v = np.zeros_like(x)
        vt = RNG.normal(size=(len(act), 2))
        vt = speed * vt / np.linalg.norm(vt, axis=1, keepdims=True)
        V = v.reshape(-1, 3)
        V[c.surf[act]] = np.einsum("kia,ka->ki", Tb, vt) + 0.05 * n[act]  # plus a normal component
        # kappa = 0 leaves only the friction terms (lambda stays lagged from begin_step)
        kap, c.kappa = c.kappa, 0.0
        G, H, _ = c.grad_hess(x, v, h, model.ndof)
        G0, H0 = 0 * G, 0 * H
        Gf = G.reshape(-1, 3)[c.surf[act]]
        Hf = H.toarray()
        y = np.linalg.norm(vt, axis=1)
        mag = np.linalg.norm(Gf, axis=1) / (c.mu * lam)
        expect = np.where(y >= c.eps_v, 1.0, sob.f1_friction(y, c.eps_v)[1])
        ok_mag.append(np.abs(mag - expect).max())
        ok_dir.append((np.einsum("ki,ki->k", -Gf, np.einsum("kia,ka->ki", Tb, vt)) / (np.linalg.norm(Gf, axis=1) * y) + 1).max())
        ok_perp.append(np.abs(np.einsum("ki,ki->k", Gf, n[act])).max() / np.abs(Gf).max())
        ok_psd.append(min(np.linalg.eigvalsh(Hf[np.ix_(r, r)])[0] / np.abs(Hf).max()
                          for r in [np.r_[3 * a: 3 * a + 3] for a in c.surf[act]]))
        diss.append(float((G - G0) @ v))
        Df = lambda u: c.phi(x, u, h) - c.phi(x, 0 * u, h)
        w = unit(RNG.normal(size=x.size))
        e_g.append(abs(fd(Df, v, w, 1e-8 * max(speed, 1e-3)) / ((G - G0) @ w) - 1))
        e_h.append(rel(fd(lambda u: c.grad_hess(x, u, h, model.ndof)[0], v, w, 1e-6 * min(speed, c.eps_v)), (H - H0) @ w))
        c.kappa = kap
    check(S, "5 friction force = grad D (FD), friction Hessian = grad of it (FD)", f"{max(e_g):.1e}, {max(e_h):.1e}",
          max(e_g) < 1e-5 and max(e_h) < 1e-4)
    check(S, "6 Coulomb: |f_t| = mu lambda f1(|u|) (= mu lambda when sliding), opposite to slip, tangential",
          f"{max(ok_mag):.1e}, cos(f_t, slip) + 1 <= {max(ok_dir):.1e}, normal part {max(ok_perp):.1e}",
          max(ok_mag) < 1e-12 and max(ok_dir) < 1e-12 and max(ok_perp) < 1e-12, "slip speeds 1e-5 .. 2 m/s, eps_v = 1e-3")
    check(S, "7 friction dissipates (grad D . v >= 0) and its Hessian is PSD without projection",
          f"min grad D.v = {min(diss):.2e}, min eig/max entry {min(ok_psd):+.1e}", min(diss) >= 0 and min(ok_psd) > -1e-12)


# ---------------------------------------------------------------------------
# E. Mesh-mesh contact
# ---------------------------------------------------------------------------
class _NoProjection:
    """ipctk stand-in that maps PSDProjectionMethod.CLAMP to NONE (for FD Hessian checks)."""

    def __init__(self, mod):
        self._m = mod

    def __getattr__(self, k):
        if k == "PSDProjectionMethod":
            return types.SimpleNamespace(CLAMP=self._m.PSDProjectionMethod.NONE)
        return getattr(self._m, k)


def section_e():
    S = "E"
    h = 1e-3
    o = replace(OPT, scene="fem", gap0=3e-4)
    model, c, sph, X0 = sob.build_scene(o)
    x = X0.ravel() + 1e-6 * RNG.normal(size=X0.size)
    G, H, nact = c.grad_hess(x, np.zeros_like(x), h, model.ndof)
    Bf = lambda y: h * c.phi(y, np.zeros_like(y), h)
    eg = max(abs(fd(Bf, x, w, 1e-9) / (G @ w) - 1) for w in [unit(RNG.normal(size=x.size)) for _ in range(3)])
    c.ipctk = _NoProjection(c.ipctk)
    try:
        _, Hn, _ = c.grad_hess(x, np.zeros_like(x), h, model.ndof)
        w = unit(RNG.normal(size=x.size))
        eh = rel(fd(lambda y: c.grad_hess(y, np.zeros_like(y), h, model.ndof)[0], x, w, 1e-9), Hn @ w / h)
    finally:
        c.ipctk = c.ipctk._m
    check(S, f"1 mesh barrier ({nact} pairs): kappa grad B = FD, kappa hess B (unprojected) = FD", f"{eg:.1e}, {eh:.1e}",
          eg < 1e-6 and eh < 1e-5)
    g3 = G.reshape(-1, 3)
    xr = x.reshape(-1, 3)
    sc = np.abs(g3).sum()
    check(S, "2 contact forces are internal: sum f = 0, sum x cross f = 0",
          f"{np.abs(g3.sum(0)).max() / sc:.1e}, {np.abs(np.cross(xr, g3).sum(0)).max() / (sc * 0.15):.1e}",
          np.abs(g3.sum(0)).max() < 1e-12 * sc and np.abs(np.cross(xr, g3).sum(0)).max() < 1e-11 * sc * 0.15)

    ipctk = c.ipctk
    worst, cons = np.inf, []
    for _ in range(3):
        dx = np.zeros_like(x).reshape(-1, 3)
        dx[sph] = np.array([0.0, 0.0, -2e-3]) + 2e-4 * RNG.normal(size=(len(sph), 3))
        dx = dx.ravel()
        a = c.max_step(x, dx)
        md = 0.1 * c.min_distance(x, net=c.dhat)
        ts = np.linspace(0, 1, 401)
        dists = np.array([c.min_distance(x + t * dx, net=c.dhat) for t in ts])
        worst = min(worst, dists[ts <= a].min() / md)
        first_bad = ts[np.argmax(dists < md)] if (dists < md).any() else 1.0
        cons.append(a / first_bad)
    check(S, "3 CCD (ipctk, min separation 0.1 x current gap): gap >= 0.1 d0 along [0, alpha]",
          f"min gap/(0.1 d0) = {worst:.3f}, alpha / first crossing = {min(cons):.2f}..{max(cons):.2f}", worst >= 1 - 1e-6,
          "ipctk CCD is conservative, so alpha stops short of the exact crossing")

    # barrier magnitude for flat parallel contact, through MeshIPC
    dh, k, gap = 1e-3, 1e4, 3e-4
    b = sob.barrier(np.array([gap]), dh)[0][0]

    def plate(n, L, z, off, rot):
        g = np.linspace(-L / 2, L / 2, n + 1)
        cs, sn = np.cos(rot), np.sin(rot)
        P = np.array([[cs * u - sn * v + off[0], sn * u + cs * v + off[1], z] for v in g for u in g])
        T = [[a, a + 1, a + n + 2] for j in range(n) for i in range(n) for a in [j * (n + 1) + i]]
        T += [[a, a + n + 2, a + n + 1] for j in range(n) for i in range(n) for a in [j * (n + 1) + i]]
        return P, np.array(T)

    out = []
    for na, nb, rot in ((32, 16, 0.0), (64, 32, 0.0), (32, 16, 0.5), (64, 32, 0.5), (32, 32, 0.5), (16, 64, 0.5)):
        Pa, Ta = plate(na, 1.0, 0.0, (0, 0), 0.0)
        Pb, Tb = plate(nb, 0.5, gap, (0.0137, 0.0091), rot)
        X = np.vstack([Pa, Pb])
        mc = sob.MeshIPC(X, np.vstack([Ta, Tb + len(Pa)]).astype(np.int32), dh, k)
        out.append((na, nb, rot, mc.phi(X.ravel(), np.zeros(X.size), 1.0) / (k * 0.25 * b)))
    check(S, "4 mesh barrier for flat parallel contact vs one-sided kappa*A*b(d)",
          "; ".join(f"{na}/{nb}{' rot' if rot else ''}: {r:.2f}" for na, nb, rot, r in out), None,
          "depends on the two meshes' relative density/orientation; analytic path is exactly 1")

    # same sphere pose, rigid analytic box vs T10 box mesh
    orr = replace(OPT, scene="rigid", contact="ipc", gap0=3e-4)
    _, cr, _, Xr = sob.build_scene(orr)
    Br = cr.phi(Xr.ravel(), np.zeros(Xr.size), 1.0)
    Bm = c.phi(X0.ravel(), np.zeros(X0.size), 1.0)
    check(S, "5 same sphere pose 0.3 mm above the box: B_mesh / B_rigid", f"{Bm / Br:.2f}", None,
          "kappa in the fem scene is effectively this much stiffer than in the rigid scene")


# ---------------------------------------------------------------------------
# F. Solver: assembled residual and Hessian are grad Phi and hess Phi
# ---------------------------------------------------------------------------
def code_g_H(model, contact, x_n, v_n, h, f_ext):
    """Run newton_step's first iteration and capture the (H, -g) it hands to the linear solver."""
    calls, xs = [], []
    el = model.elastic

    def el_wrap(x, need_hess=True):
        if need_hess:
            xs.append(x.copy())
        return el(x, need_hess)

    def spsolve(A, b):
        calls.append((A.copy(), b.copy()))
        return spla.spsolve(A, b)

    model.elastic, sob.spla = el_wrap, types.SimpleNamespace(spsolve=spsolve)
    try:
        sob.newton_step(model, contact, x_n, v_n, h, f_ext, replace(OPT, max_newton=1, line_search=False, adaptive_kappa=False),
                        sob.StepStats())
    finally:
        del model.elastic
        sob.spla = spla
    A, b = calls[0]
    return (xs[0] - x_n) / h, -b, A


def Phi(model, contact, x_n, v_n, h, f_ext, v):
    """The step energy as written in the write-up, from the energy functions only."""
    x = x_n + h * v
    dv = v - v_n
    return 0.5 / h * dv @ (model.M @ dv) + (model.elastic(x, need_hess=False)[0] - f_ext @ x) / h + contact.phi(x, v, h)


def section_f():
    S = "F"
    h = 1e-3
    o, model0, c, X0 = rigid_state(5e-4, 4e-3, 0.3)
    model = sob.T10Model(X0, model0.T, o.E, o.nu, o.rho, project_elastic=False)
    x_n = X0.ravel() + 1e-5 * RNG.normal(size=X0.size)
    v_n = np.tile([0.02, -0.01, -0.05], model.n) + 1e-3 * RNG.normal(size=X0.size)
    f_ext = model.gravity(np.array([0, 0, -9.81]))
    c.begin_step(x_n, v_n)
    v0, g, H = code_g_H(model, c, x_n, v_n, h, f_ext)
    P = lambda v: Phi(model, c, x_n, v_n, h, f_ext, v)
    eg = max(abs(fd(P, v0, w, 1e-6) / (g @ w) - 1) for w in [unit(RNG.normal(size=v0.size)) for _ in range(3)])
    check(S, "1 assembled residual g = grad_v Phi (elastic + inertia + gravity + barrier + friction)",
          f"{eg:.1e} (start v = v_n: {np.abs(v0 - v_n).max():.0e})", eg < 1e-6)
    eh = []
    for _ in range(2):
        w = unit(RNG.normal(size=v0.size))
        e = 1e-6
        gp = code_g_H(model, c, x_n, v_n + e * w, h, f_ext)[1] + model.M @ (e * w) / h
        gm = code_g_H(model, c, x_n, v_n - e * w, h, f_ext)[1] - model.M @ (e * w) / h
        eh.append(rel((gp - gm) / (2 * e), H @ w))
    check(S, "2 assembled Hessian H = dg/dv = M/h + h K_t + h kappa hess B + hess D", f"{max(eh):.1e}", max(eh) < 1e-5,
          "projections off: unprojected K_t, face contact")

    # full solve with the production settings: converges to a stationary point of Phi
    o, model, c, X0 = rigid_state(5e-4, 4e-3, 0.3)
    v_n = np.tile([0.1, 0.0, -0.8], model.n)
    c.begin_step(x_n, v_n)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st = sob.StepStats()
        v = sob.newton_step(model, c, x_n, v_n, h, f_ext, replace(OPT, verbose_newton=True), st)
    seq = [float(s) for s in re.findall(r"\|dv\|inf=(\S+)", buf.getvalue())]
    al = [float(s) for s in re.findall(r"alpha=(\S+)", buf.getvalue())]
    x = x_n + h * v
    gc, _, nact = c.grad_hess(x, v, h, model.ndof)
    g_end = model.M @ (v - v_n) / h + model.elastic(x, need_hess=False)[1] - f_ext + gc
    v_s = sob.newton_step(model, c, x_n, v_n, h, f_ext, replace(OPT, max_newton=0), sob.StepStats())
    gs, _, _ = c.grad_hess(x_n + h * v_s, v_s, h, model.ndof)
    g_start = model.M @ (v_s - v_n) / h + model.elastic(x_n + h * v_s, need_hess=False)[1] - f_ext + gs
    check(S, f"3 Newton converges to grad Phi = 0 ({st.newton} its, {nact} active, CCD min alpha {st.ccd_min_alpha:.2f})",
          f"|g_end|/|g_start| = {np.linalg.norm(g_end) / np.linalg.norm(g_start):.1e}",
          st.converged and np.linalg.norm(g_end) < 1e-6 * np.linalg.norm(g_start))
    tail = seq[-4:]
    check(S, "4 Newton convergence rate", "|dv|inf per iteration: " + ", ".join(f"{s:.1e}" for s in seq)
          + "; alpha: " + ", ".join(f"{a:.2g}" for a in al), None,
          "quadratic tail: " + ", ".join(f"{np.log(tail[i + 1]) / np.log(tail[i]):.2f}" for i in range(len(tail) - 1)) + " (ratio of logs ~2)")


# ---------------------------------------------------------------------------
# G. Time stepping and conservation
# ---------------------------------------------------------------------------
def run_captured(opt, scene=None, on_contact=None):
    """sob.run with the scene objects and every frame exposed."""
    orig = sob.build_scene
    got = {}

    def bs(o):
        out = (scene or orig)(o)
        got.update(model=out[0], contact=out[1], sph=out[2], X0=out[3])
        if on_contact:
            on_contact(out[1])
        return out

    with tempfile.TemporaryDirectory() as td:
        sob.build_scene = bs
        try:
            s, hist = sob.run(replace(opt, record=os.path.join(td, "r.npz")))
        finally:
            sob.build_scene = orig
        got["frames"] = np.load(os.path.join(td, "r.npz"))["frames"].reshape(len(hist) + 1, -1)
    return s, hist, got


def velocities(frames, v0, h, bdf2):
    v = [v0]
    for k in range(1, len(frames)):
        if bdf2 and k > 1:
            v.append((3 * frames[k] - 4 * frames[k - 1] + frames[k - 2]) / (2 * h))
        else:
            v.append((frames[k] - frames[k - 1]) / h)
    return np.array(v)


def section_g():
    S = "G"
    g, h = 9.81, 1e-3
    for integ in ("be", "bdf2"):
        o = replace(OPT, scene="rigid", contact="ipc", gravity=g, v0=0.0, gap0=1.0, dt=h, t_end=0.02, integrator=integ)
        _, hist, got = run_captured(o)
        n = np.arange(1, len(hist) + 1)
        z0 = (got["model"].node_mass * got["X0"][:, 2]).sum() / got["model"].node_mass.sum()
        vz = np.array([r["com_vz"] for r in hist])
        z = np.array([r["com_z"] for r in hist])
        if integ == "be":
            z_ex = z0 - g * h * h * n * (n + 1) / 2
        else:
            z_ex = z0 - 0.5 * g * (n * h) ** 2 - 0.75 * g * h * h * (1 - 3.0 ** (-n))
        ev, ez = np.abs(vz + g * n * h).max() / (g * n[-1] * h), np.abs(z - z_ex).max() / (0.5 * g * (n[-1] * h) ** 2)
        check(S, f"1 free fall, {integ.upper()}: v_n = -g t_n, z_n = discrete closed form", f"{ev:.1e}, {ez:.1e}",
              max(ev, ez) < 1e-9, "BE: z0 - g h^2 n(n+1)/2; BDF2: exact parabola - 3/4 g h^2 (1 - 3^-n) from the BE first step")

    # momentum: sphere A (moving) hits free sphere B (at rest), oblique, with friction, no gravity, no fixed nodes
    def two_spheres(o):
        Xs, Ts = sphere()
        R = np.linalg.norm(Xs, axis=1).max()
        n = len(Xs)
        X = np.vstack([Xs, Xs + np.array([0.05, 0.0, 2 * R + o.gap0])])
        T = np.vstack([Ts, Ts + n])
        model = sob.T10Model(X, T, o.E, o.nu, o.rho)
        contact = sob.MeshIPC(X, sob.surface_triangles_4way(T).astype(np.int32), o.dhat, o.kappa, o.mu, o.eps_v)
        return model, contact, n + np.arange(n), X

    inter = []

    def watch_mesh(c):
        gh = c.grad_hess

        def wrapped(x, v, hh, ndof):
            inter.append(bool(c.ipctk.has_intersections(c.cm, c._V(x))))
            return gh(x, v, hh, ndof)

        c.grad_hess = wrapped

    o = replace(OPT, scene="fem", v0=1.0, gap0=2e-3, mu=0.3, dt=h, t_end=0.014, integrator="bdf2")
    t0 = time.time()
    s, hist, got = run_captured(o, two_spheres, watch_mesh)
    m, sph = got["model"], got["sph"]
    v0 = np.zeros(m.ndof)
    v0.reshape(-1, 3)[sph] = [0, 0, -1.0]
    V = velocities(got["frames"], v0, h, True)
    Pm = np.array([(m.M @ v).reshape(-1, 3).sum(0) for v in V])
    nB = len(sph)
    PB = np.array([(m.M @ v).reshape(-1, 3)[:nB].sum(0) for v in V])
    xs = got["frames"].reshape(len(V), -1, 3)
    L = np.array([np.cross(x, (m.M @ v).reshape(-1, 3)).sum(0) for x, v in zip(xs, V)])
    p0 = np.linalg.norm(Pm[0])
    act = max(r["active"] for r in hist)
    check(S, f"2 linear momentum, two free spheres colliding (BDF2, mu 0.3, {act} contact pairs max)",
          f"max |P_n - P_0| / |P_0| = {np.abs(Pm - Pm[0]).max() / p0:.1e}; sphere B picked up {PB[-1] / p0} x P_0",
          np.abs(Pm - Pm[0]).max() < 1e-6 * p0 and act > 0, f"{time.time() - t0:.0f} s")
    check(S, "3 angular momentum about the origin, same run", f"max |L_n - L_0| / (|P_0| R) = {np.abs(L - L[0]).max() / (p0 * 0.15):.1e}",
          None, "BE/BDF2 conserve it only to O(h); not a contact property")
    check(S, "4 every accepted Newton iterate intersection-free (mesh-mesh, ipctk exact test)",
          f"{sum(inter)} of {len(inter)} iterates intersecting", len(inter) > 0 and not any(inter))

    # energy: rigid impact
    for contact, integ in (("ipc", "be"), ("ipc", "bdf2"), ("penalty", "be")):
        dmins = []

        def watch_rigid(c):
            if contact != "ipc":
                return
            gh = c.grad_hess

            def wrapped(x, v, hh, ndof):
                dmins.append(c.min_distance(x))
                return gh(x, v, hh, ndof)

            c.grad_hess = wrapped

        o = replace(OPT, scene="rigid", contact=contact, dt=h, t_end=0.03, integrator=integ)
        _, hist, got = run_captured(o, None, watch_rigid)
        c = got["contact"]
        if contact == "ipc":
            U = [c.phi(x, np.zeros_like(x), 1.0) for x in got["frames"]]
        else:
            U = [0.5 * c.k * (c.A * np.maximum(-c.box.signed_distance(x.reshape(-1, 3)[c.surf])[0], 0) ** 2).sum() for x in got["frames"]]
        E0 = 0.5 * got["model"].node_mass.sum() * o.v0**2
        En = np.array([E0] + [r["KE"] + r["Psi"] for r in hist]) + np.array(U)
        inc = np.diff(En).max() / E0
        label = f"{contact} {integ.upper()}"
        if contact == "ipc" and integ == "be":
            check(S, f"5 energy, rigid impact {label}: KE + Psi + kappa B never increases",
                  f"max step increase {inc:+.1e} E0, E_end/E0 = {En[-1] / E0:.3f}", inc < 1e-6)
            check(S, "6 every accepted Newton iterate has gap > 0 (rigid, 1 m/s impact)",
                  f"min over {len(dmins)} iterates = {min(dmins):.2e} m", min(dmins) > 0)
        else:
            check(S, f"5 energy, rigid impact {label}", f"max step increase {inc:+.1e} E0, E_end/E0 = {En[-1] / E0:.3f}", None,
                  "BDF2 is not monotone step to step" if contact == "ipc" else "penalty energy includes 1/2 k A delta^2")


def main():
    t0 = time.time()
    Xs, Ts = section_a()
    section_b(Xs, Ts)
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    tags = Counter(r[3] for r in ROWS)
    print(f"\n{tags['PASS']} pass, {tags['FAIL']} fail, {tags['NOTE']} notes in {time.time() - t0:.0f} s")
    sys.exit(1 if tags["FAIL"] else 0)


if __name__ == "__main__":
    main()
