#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""
IPC version of lib_bin/ptest/brick_sliding.cc: a 1 kg T10 brick released at rest on a fixed
slope. Same brick mesh, slope angles, placement, material, dt and step count as the DEME demo;
contact is the prototype's IPC barrier + lagged smoothed Coulomb friction (one mu).

The slope is fixed in the demo (constraint_type 0: every slope node fixed), so it is modelled
as a rigid half-space whose plane is taken from the slope mesh the same way the demo does.

    uv run test-scripts/ipc_proto/brick_slope.py                     # all slopes, mu = 0.25 and 0.2
    uv run test-scripts/ipc_proto/brick_slope.py --slopes 04 --mus 0.25
    uv run test-scripts/ipc_proto/brick_slope.py --plot --deme <dir with brick_sliding outputs>
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sphere_on_box as sob  # noqa: E402

G = 9.81
BRICK = os.path.join(sob.REPO, "data/meshes/ptest/brick/brick_medium.1")
SLOPE = os.path.join(sob.REPO, "data/meshes/ptest/slope/slope_{}.1")
# brick_sliding.cc slope angles (rad)
ALPHA = {"01": 0.18, "02": 0.197395560, "03": 0.244978663, "03p1": 0.2440372654, "03p2": 0.2459196179,
         "03p3": float(np.arctan(0.247)), "03p4": float(np.arctan(0.248)), "04": 0.25}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "brick_slope")


class RigidPlane:
    """Half-space n.p <= b, with sob.RigidBox's interface (distance to a convex set)."""

    def __init__(self, n, b):
        self.n, self.b = n / np.linalg.norm(n), b

    def distance(self, p):
        return np.maximum(p @ self.n - self.b, 0.0), np.tile(self.n, (len(p), 1)), np.zeros((len(p), 3, 3))

    def signed_distance(self, p):
        return p @ self.n - self.b, np.tile(self.n, (len(p), 1))


class SlopeMeshIPC(sob.MeshIPC):
    """sob.MeshIPC with the slope's surface triangles as fixed collision geometry: the slope
    vertices sit after the brick nodes in the collision mesh, never move, and carry no DOFs."""

    def __init__(self, Xb, tris_b, Xs, tris_s, dhat, kappa, mu, eps_v):
        self.nb, self.Xs = len(Xb), Xs
        tris = np.vstack([tris_b, tris_s + len(Xb)]).astype(np.int32)
        super().__init__(np.vstack([Xb, Xs]), tris, dhat, kappa, mu, eps_v)

    def _V(self, x):
        return self.cm.vertices(np.asfortranarray(np.vstack([x.reshape(-1, 3), self.Xs])))

    def _vel(self, v):
        return self.cm.map_displacements(np.asfortranarray(np.vstack([v.reshape(-1, 3), np.zeros_like(self.Xs)])))

    def grad_hess(self, x, v, h, ndof):
        G, H, n_act = super().grad_hess(x, v, h, ndof)
        m = 3 * self.nb
        return G[:m], H[:m, :m], n_act

    def min_distance(self, x, net=1e-3):
        # a 1 mm search net is plenty for the recorded gap and keeps the broad phase small
        return super().min_distance(x, net=min(net, 1e-3))


def slope_top_surface(key, n, b, tol=1e-7):
    """Slope surface triangles (4-way split T10 faces) lying on the top plane n.p = b, compacted."""
    Xs, Ts = sob.read_tetgen_t10(SLOPE.format(key))
    tris = sob.surface_triangles_4way(Ts)
    tris = tris[np.all(np.abs(Xs[tris] @ n - b) < tol, axis=1)]
    used, inv = np.unique(tris, return_inverse=True)
    return Xs[used], inv.reshape(tris.shape).astype(np.int32)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rest_gap(p, kappa, dhat):
    """Gap at which the barrier pressure -kappa b'(d) equals p."""
    lo, hi = 1e-12, dhat
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if -kappa * sob.barrier(np.array([mid]), dhat)[1][0] > p:
            lo = mid
        else:
            hi = mid
    return mid


def brick_scene(opt, key):
    alpha = ALPHA[key]
    n = np.array([-np.tan(alpha), 0.0, 1.0])
    n /= np.linalg.norm(n)
    Xb, Tb = sob.read_tetgen_t10(BRICK)
    X = Xb @ rot_y(-alpha).T + np.array([0.8, 0.0, 0.0])  # as brick_sliding.cc
    # plane offset: highest slope node (along n) under the brick footprint, as the demo does
    Xs, _ = sob.read_tetgen_t10(SLOPE.format(key))
    lo, hi = X[:, :2].min(0) - 0.02, X[:, :2].max(0) + 0.02
    under = ((Xs[:, :2] >= lo) & (Xs[:, :2] <= hi)).all(1)
    b = float((Xs[under] @ n).max())
    vol = sob.T10Model(X, Tb, opt.E, opt.nu, 1.0).node_mass.sum()
    model = sob.T10Model(X, Tb, opt.E, opt.nu, 1.0 / vol)  # 1 kg brick
    tris = sob.surface_triangles_4way(Tb)
    area = sob.vertex_areas(X, tris, len(X))
    bottom = np.unique(tris[np.all(np.abs(X[tris] @ n - (X @ n).min()) < 1e-9, axis=1)])
    p = G * np.cos(alpha) / area[bottom].sum()
    # at rest under full gravity; with a gravity ramp start at dhat (no contact force yet)
    gap = 0.999 * opt.dhat if opt.gravity_ramp > 0 else rest_gap(p, opt.kappa, opt.dhat)
    X = X + ((b + gap) - (X @ n).min()) * n
    model = sob.T10Model(X, Tb, opt.E, opt.nu, 1.0 / vol)
    if opt.slope_mesh:
        assert opt.mu_s <= 0, "static/kinetic friction is only implemented for the plane"
        Xs, tris_s = slope_top_surface(key, n, b)
        contact = SlopeMeshIPC(X, tris, Xs, tris_s, opt.dhat, opt.kappa, opt.mu, opt.eps_v)
        return model, contact, np.arange(len(X)), X
    contact = sob.RigidBoxIPC(RigidPlane(n, b), np.unique(tris), area, opt.dhat, opt.kappa, opt.mu, opt.eps_v,
                              mu_s=opt.mu_s if opt.mu_s > 0 else None)
    return model, contact, np.arange(len(X)), X


def analytic(alpha, mu, t, mu_s=-1.0, ramp=0.0, t_ramp=0.0):
    """Rigid block released at rest (rigid-block Coulomb with static mu_s, kinetic mu).
    ramp: gravity ramped from 0 over `ramp` s. t_ramp: full normal load, along-slope load ramped
    over `t_ramp` s (tilt test). Sticks while the along-slope load <= mu_s x normal load (a load
    ratio within 1e-6 of mu_s counts as stick: slope 02 is tan a = 0.2 to 1e-9)."""
    mu_s = max(mu, mu_s)
    tf = np.arange(0.0, float(np.max(t)) + 1e-5, 1e-5)
    rg = np.minimum(1.0, tf / ramp) if ramp > 0 else np.ones_like(tf)
    rt = np.minimum(1.0, tf / t_ramp) if t_ramp > 0 else np.ones_like(tf)
    tan_load = G * np.sin(alpha) * rg * rt
    nrm_load = G * np.cos(alpha) * rg
    v, s, vs, ss, sliding = 0.0, 0.0, [0.0], [0.0], False
    for k in range(1, len(tf)):
        dt = tf[k] - tf[k - 1]
        if not sliding and tan_load[k] > (mu_s + 1e-6) * nrm_load[k]:
            sliding = True
        a = max(tan_load[k] - mu * nrm_load[k], 0.0 if v <= 0 else -np.inf) if sliding else 0.0
        v = max(v + a * dt, 0.0)
        s += v * dt
        vs.append(v)
        ss.append(s)
    return np.interp(t, tf, ss), np.interp(t, tf, vs)


def run_case(args):
    key, mu, base, tag = args
    opt = replace(base, mu=mu, load_normal=(-np.sin(ALPHA[key]), 0.0, np.cos(ALPHA[key])))
    alpha = ALPHA[key]
    down = np.array([-np.cos(alpha), 0.0, -np.sin(alpha)])  # downhill along the slope
    n = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    sob.build_scene = lambda o: brick_scene(o, key)
    rec = os.path.join(OUT, f"rec{tag}_s{key}_mu{mu}.npz")
    t0 = time.time()
    summ, hist = sob.run(replace(opt, record=rec))
    d = np.load(rec)
    frames, X0 = d["frames"], d["X0"]
    w = sob.T10Model(X0, d["T"], opt.E, opt.nu, 1.0).node_mass
    com = (frames * w[None, :, None]).sum(1) / w.sum()
    t = d["t"]
    s = (com - com[0]) @ down
    v = np.r_[0.0, [np.array([r["com_vx"], 0.0, r["com_vz"]]) @ down for r in hist]]
    off = (com - com[0]) @ n
    ax0 = X0[np.argmax(X0 @ -down)] - X0[np.argmax(X0 @ down)]  # brick long axis
    ax = frames[:, np.argmax(X0 @ -down)] - frames[:, np.argmax(X0 @ down)]
    rot = np.degrees(np.arccos(np.clip(ax @ ax0 / (np.linalg.norm(ax, axis=1) * np.linalg.norm(ax0)), -1, 1)))
    s_ref, v_ref = analytic(alpha, mu, t, opt.mu_s, opt.gravity_ramp, opt.tangent_ramp)
    out = dict(
        slope=key, alpha=alpha, tan_alpha=float(np.tan(alpha)), mu=mu, t=t.tolist(), s=s.tolist(), v=v.tolist(),
        s_ref=s_ref.tolist(), v_ref=v_ref.tolist(), normal_offset=off.tolist(), rot_deg=rot.tolist(),
        dmin=[r["dmin"] for r in hist], newton=[r["newton"] for r in hist], kappa_end=summ["kappa_end"],
        unconverged=summ["unconverged"], wall_s=round(time.time() - t0, 1),
    )
    out.update(mu_s=opt.mu_s, eps_v=opt.eps_v, gravity_ramp=opt.gravity_ramp, tangent_ramp=opt.tangent_ramp, tag=tag)
    json.dump(out, open(os.path.join(OUT, f"ipc{tag}_s{key}_mu{mu}.json"), "w"))
    return (f"slope {key:5s} tan {np.tan(alpha):.4f} mu {mu:.2f}: s(0.3 s) = {s[-1] * 1e3:8.3f} mm "
            f"(analytic {s_ref[-1] * 1e3:7.3f}), v = {v[-1] * 1e3:7.2f} mm/s, rot {rot.max():.2e} deg, "
            f"min gap {min(out['dmin']) * 1e6:.1f} um, newton {np.mean(out['newton']):.1f}/step, {out['wall_s']} s")


def load_deme(root, key):
    """rigid_equiv.csv from a brick_sliding run for this slope, if present."""
    hits = sorted(glob.glob(os.path.join(root, f"*_s{key}_*", "rigid_equiv*.csv")))
    if not hits:
        return None
    rows = np.genfromtxt(hits[0], delimiter=",", names=True, dtype=None, encoding="utf-8")
    return rows[rows["body"] == "brick"]


def plot(deme_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(ALPHA, key=lambda k: ALPHA[k])
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
    for ax, key in zip(axes.ravel(), keys):
        alpha = ALPHA[key]
        down = np.array([-np.cos(alpha), 0.0, -np.sin(alpha)])
        for mu, col in ((0.25, "#2563eb"), (0.2, "#7c3aed")):
            f = os.path.join(OUT, f"ipc_s{key}_mu{mu}.json")
            if os.path.exists(f):
                r = json.load(open(f))
                t = np.array(r["t"])
                ax.plot(t, np.array(r["s"]) * 1e3, color=col, lw=2, label=f"IPC mu={mu}")
                ax.plot(t, np.array(r["s_ref"]) * 1e3, color=col, lw=1, ls="--", label=f"analytic mu={mu}")
        if deme_root:
            rows = load_deme(deme_root, key)
            if rows is not None:
                com = np.column_stack([rows["com_x"], rows["com_y"], rows["com_z"]])
                ax.plot(rows["time"], (com - com[0]) @ down * 1e3, color="#dc2626", lw=2,
                        label="DEME (mu_s 0.25, mu_k 0.2)")
        ax.set_title(f"slope {key}: tan a = {np.tan(alpha):.4f}", fontsize=10)
        ax.set_ylabel("slid down slope (mm)")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("t (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("1 kg T10 brick on a fixed slope: IPC prototype vs analytic" + (" vs DEME" if deme_root else ""))
    fig.tight_layout()
    path = os.path.join(OUT, "brick_slope.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slopes", default=",".join(ALPHA))
    p.add_argument("--mus", default="0.25,0.2")
    p.add_argument("--dhat", type=float, default=1e-4)
    p.add_argument("--kappa", type=float, default=25.0, help="Pa; resting gap ~10 um at the brick's ~480 Pa")
    p.add_argument("--integrator", default="be")
    p.add_argument("--mu-s", type=float, default=-1.0, help="> 0: static/kinetic law, mu_k = each --mus value")
    p.add_argument("--eps-v", type=float, default=1e-3)
    p.add_argument("--gravity-ramp", type=float, default=0.0, help="s; ramp gravity from 0")
    p.add_argument("--tangent-ramp", type=float, default=0.0,
                   help="s; full normal gravity from t = 0, along-slope component ramped (tilt test)")
    p.add_argument("--slope-mesh", action="store_true",
                   help="collide with the slope's top-face triangles (ipctk mesh contact) instead of a plane")
    p.add_argument("--tag", default="", help="suffix for output file names, e.g. _sk_eps1e-4")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--plot", action="store_true", help="only plot existing results")
    p.add_argument("--deme", default="", help="directory holding brick_sliding output folders")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        plot(a.deme)
        return
    base = sob.Options(scene="brick", contact="ipc", dt=5e-4, t_end=0.3, v0=0.0, gravity=G, E=1e7, nu=0.3,
                       dhat=a.dhat, kappa=a.kappa, eps_v=a.eps_v, mu_s=a.mu_s, gravity_ramp=a.gravity_ramp, tangent_ramp=a.tangent_ramp, slope_mesh=a.slope_mesh, lag_iters=1, integrator=a.integrator, quiet=True)
    cases = [(k, float(mu), base, a.tag) for k in a.slopes.split(",") for mu in a.mus.split(",")]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(cases))) as ex:
        for line in ex.map(run_case, cases):
            print(line, flush=True)
    if not a.tag:
        plot(a.deme)


if __name__ == "__main__":
    main()
