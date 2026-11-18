#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "ipctk==1.6.0", "gmsh"]
# ///
"""
Quasi-static Hertz check for the IPC prototype.

A quarter T10 hemisphere (radius R, refined at the pole, symmetry BCs on x=0 and y=0)
is pressed onto a rigid plane by prescribing the z displacement of its flat top face.
Each load step is a static IPC solve:

    minimize  Psi(x) + kappa B(x)   over free DOFs,  CCD-capped energy line search

and the contact pressure on the sphere surface nodes is compared with Hertz:

    a = (3 F R / 4 E*)^(1/3),  p0 = 3 F / (2 pi a^2),  p(r) = p0 sqrt(1 - r^2/a^2)

Contact variants
    analytic   sphere surface nodes vs the exact plane (RigidBoxIPC from sphere_on_box.py)
    convergent ipc-toolkit, triangulated plane, area weighting + improved max approximation
    plain      ipc-toolkit, triangulated plane, original IPC pair sum (no area weighting)

Examples
    uv run test-scripts/ipc_proto/hertz.py --levels 3,6 --variants analytic,convergent
    uv run test-scripts/ipc_proto/hertz.py --out results_hertz/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(__file__))
import sphere_on_box as sob  # noqa: E402

R = 0.15
E, NU, RHO = 1e7, 0.3, 70.7355
E_STAR = E / (1 - NU**2)  # rigid plane


def hertz(F):
    a = (3 * F * R / (4 * E_STAR)) ** (1 / 3)
    return a, 3 * F / (2 * np.pi * a * a), a * a / R


# 6-node triangle mass matrix / (A/180), nodes (c0, c1, c2, m01, m12, m20).
# Corner rows sum to 0: a quadratic face carries no uniform-pressure load at its corners.
M6 = np.array([[6, -1, -1, 0, -4, 0],
               [-1, 6, -1, 0, 0, -4],
               [-1, -1, 6, -4, 0, 0],
               [0, 0, -4, 32, 16, 16],
               [-4, 0, 0, 16, 32, 16],
               [0, -4, 0, 16, 16, 32]], float) / 180.0


def boundary_faces_t10(T):
    """Boundary faces of a T10 mesh as (c0, c1, c2, m01, m12, m20) node indices."""
    slot = {frozenset(e): 4 + k for k, e in enumerate(sob.EDGES)}
    seen = {}
    for e, t in enumerate(T):
        for f in sob.FACES:
            seen.setdefault(tuple(sorted(t[list(f)])), []).append((e, f))
    out = []
    for owners in seen.values():
        if len(owners) == 1:
            e, f = owners[0]
            t = T[e]
            out.append([t[f[0]], t[f[1]], t[f[2]], t[slot[frozenset((f[0], f[1]))]],
                        t[slot[frozenset((f[1], f[2]))]], t[slot[frozenset((f[2], f[0]))]]])
    return np.array(out)


def consistent_pressure(X, faces, nodes, f_n):
    """Pressure field p (quadratic on each T10 face) whose consistent nodal loads equal f_n.

    Solves M_s p = f_n with M_s the surface mass matrix of the 6-node faces. Dividing f_n by a
    lumped area instead ignores that corner nodes of quadratic faces carry almost no load."""
    import scipy.sparse as sp

    pos = -np.ones(len(X), int)
    pos[nodes] = np.arange(len(nodes))
    c = X[faces[:, :3]]
    A = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
    loc = pos[faces]
    rows = np.repeat(loc, 6, axis=1).ravel()
    cols = np.tile(loc, (1, 6)).ravel()
    vals = (A[:, None, None] * M6).ravel()
    Ms = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    return spla.spsolve(Ms.tocsc(), f_n)


# ---------------------------------------------------------------------------
# Meshes (gmsh): quarter lower hemisphere, pole at z = z_pole; quarter plane at z = 0
# ---------------------------------------------------------------------------
def _size_field(gmsh, expr):
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", expr)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0)):
        gmsh.option.setNumber(k, v)


def mesh_quarter_hemisphere(hc, hmax, grad, z_pole, r_fine):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("qhemi")
    occ = gmsh.model.occ
    s = occ.addSphere(0, 0, 0, R)
    b = occ.addBox(0, 0, -R, R, R, R)
    occ.intersect([(3, s)], [(3, b)])
    occ.synchronize()
    _size_field(gmsh, f"min({hmax}, {hc} + {grad}*max(0, sqrt(x^2 + y^2 + (z + {R})^2) - {r_fine}))")
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)  # straight-sided T10, like the TetGen meshes
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    _, ntags = gmsh.model.mesh.getElementsByType(11)
    gmsh.finalize()
    X = coords.reshape(-1, 3)
    pos = {t: i for i, t in enumerate(tags)}
    T = np.array([pos[t] for t in ntags]).reshape(-1, 10)
    used = np.unique(T)
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    X, T = X[used], remap[T]
    X[:, 2] += R + z_pole
    return X, sob.standardize_t10(X, T)


def mesh_quarter_plane(hc, hmax, grad, L, r_fine):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("qplane")
    gmsh.model.occ.addRectangle(0, 0, 0, L, L)
    gmsh.model.occ.synchronize()
    _size_field(gmsh, f"min({hmax}, {hc} + {grad}*max(0, sqrt(x^2 + y^2) - {r_fine}))")
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    _, ntags = gmsh.model.mesh.getElementsByType(2)
    gmsh.finalize()
    X = coords.reshape(-1, 3)
    pos = {t: i for i, t in enumerate(tags)}
    F = np.array([pos[t] for t in ntags]).reshape(-1, 3)
    used = np.unique(F)
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    return X[used], remap[F].astype(np.int32)


# ---------------------------------------------------------------------------
# Static IPC solve with prescribed DOFs
# ---------------------------------------------------------------------------
def static_solve(model, contact, x, free, tol, max_it=60):
    idx = np.nonzero(free)[0]
    zero = np.zeros_like(x)

    def energy(y):
        return model.elastic(y, need_hess=False)[0] + contact.phi(y, zero, 1.0)

    for it in range(max_it):
        _, f_int, K = model.elastic(x)
        G, Hc, _ = contact.grad_hess(x, zero, 1.0, model.ndof)
        g = f_int + G
        H = (K + Hc).tocsr()
        dx = np.zeros_like(x)
        dx[idx] = spla.spsolve(H[idx][:, idx].tocsc(), -g[idx])
        if np.abs(dx).max() < tol:
            return x, it + 1, True
        alpha = min(1.0, contact.max_step(x, dx))
        e0, slope = energy(x), g @ dx
        while energy(x + alpha * dx) > e0 + 1e-4 * alpha * slope and alpha > 1e-12:
            alpha *= 0.5
        x = x + alpha * dx
    return x, max_it, False


def run_case(level, variant, opt):
    a_ref = opt.a_ref
    hc = a_ref / level
    r_fine = opt.fine_radius * a_ref
    Xs, Ts = mesh_quarter_hemisphere(hc, opt.hmax, opt.grad, opt.gap0, r_fine)
    ns = len(Xs)
    tris = sob.surface_triangles_4way(Ts)
    on = lambda v, ax, val: np.abs(Xs[v, ax] - val) < 1e-9
    z_top = Xs[:, 2].max()
    drop = np.array([all(on(t, 0, 0.0)) or all(on(t, 1, 0.0)) or all(on(t, 2, z_top)) for t in tris])
    tris = tris[~drop]  # contact surface = curved part only (no symmetry or loading faces)
    area = sob.vertex_areas(Xs, tris, ns)
    surf = np.unique(tris)
    faces = curved_faces(Xs, Ts)
    assert np.array_equal(np.unique(faces), surf)

    if variant == "analytic":
        X = Xs
        model = sob.T10Model(X, Ts, E, NU, RHO, project_elastic=True)
        box = sob.RigidBox([-1.0, -1.0, -1.0], [1.0, 1.0, 0.0])
        contact = sob.RigidBoxIPC(box, surf, area, opt.dhat, opt.kappa)
        n_plane = 0
    else:
        Xp, Fp = mesh_quarter_plane(hc, opt.hmax, opt.grad, opt.plane_L, r_fine)
        X = np.vstack([Xs, Xp])
        n_plane = len(Xp)
        model = sob.T10Model(X, Ts, E, NU, RHO, fixed_nodes=np.arange(ns, ns + n_plane), project_elastic=True)
        all_tris = np.vstack([tris, Fp + ns]).astype(np.int32)
        contact = sob.MeshIPC(X, all_tris, opt.dhat, opt.kappa)
        if variant == "plain":
            ipctk = contact.ipctk

            def plain_collisions(V, c=contact):
                col = ipctk.NormalCollisions()
                col.build(c.cm, V, c.dhat)
                return col

            contact._collisions = plain_collisions
            # Original IPC: unit weight per pair. kappa is fixed across mesh levels, matched to
            # the physical barrier for one pair at the coarsest level's vertex area, so the
            # effect of refinement on an unweighted pair sum shows up.
            A_ref = np.sqrt(3) / 2 * (opt.a_ref / 3 / 2) ** 2
            contact.B = ipctk.BarrierPotential(opt.dhat, opt.kappa * A_ref / opt.dhat**3, False)

    # Boundary conditions: symmetry planes, prescribed z on the top face.
    free = model.free.copy()
    for v in range(ns):
        if abs(Xs[v, 0]) < 1e-9:
            free[3 * v] = False
        if abs(Xs[v, 1]) < 1e-9:
            free[3 * v + 1] = False
    top = np.nonzero(np.abs(Xs[:, 2] - z_top) < 1e-9)[0]
    free[3 * top + 2] = False
    model.free = free

    x = X.ravel().copy()
    sph_z = 3 * np.arange(ns) + 2
    steps = []
    frames = [Xs.copy()]
    delta_prev = 0.0
    t0 = time.time()
    for delta in np.linspace(0.0, opt.gap0 + opt.indent, opt.n_steps + 1)[1:]:
        # predictor: translate the sphere down by the increment, as far as CCD allows
        dx = np.zeros_like(x)
        dx[sph_z] = -(delta - delta_prev)
        dx[~free] = 0.0
        dx[3 * top + 2] = -(delta - delta_prev)
        alpha = contact.max_step(x, dx)
        x = x + alpha * dx
        x[3 * top + 2] = X[top, 2] - delta
        x, its, ok = static_solve(model, contact, x, free, opt.tol)
        delta_prev = delta
        G, _, n_act = contact.grad_hess(x, np.zeros_like(x), 1.0, model.ndof)
        fc = -G.reshape(-1, 3)[:ns]  # contact force on sphere nodes
        F_q = fc[surf, 2].sum()
        F = 4.0 * F_q
        xs = x.reshape(-1, 3)[:ns]
        r = np.hypot(xs[surf, 0], xs[surf, 1])
        p = fc[surf, 2] / area[surf]
        rec = dict(delta=delta, F=F, newton=its, converged=ok, active=n_act,
                   dmin=contact.min_distance(x))
        if F > 0:
            aH, p0H, dH = hertz(F)
            fz = fc[surf, 2]
            rec.update(a_eff=float(np.sqrt(2.5 * (fz * r * r).sum() / fz.sum())), p_max=float(p.max()),
                       a_hertz=aH, p0_hertz=p0H, approach=delta - opt.gap0, approach_hertz=dH)
        steps.append(rec)
        frames.append(x.reshape(-1, 3)[:ns].copy())
        if not opt.quiet:
            print(f"  [{variant} L{level}] delta={delta * 1e3:.3f} mm F={F:8.2f} N newton={its} "
                  + (f"a={rec['a_eff'] * 1e3:.2f}/{rec['a_hertz'] * 1e3:.2f} mm "
                     f"p={rec['p_max'] / 1e3:.0f}/{rec['p0_hertz'] / 1e3:.0f} kPa" if F > 0 else ""), flush=True)
    p_cons = consistent_pressure(Xs, faces, surf, fc[surf, 2])
    profile = dict(r=r.tolist(), p=p.tolist(), p_cons=p_cons.tolist())
    if opt.record:
        os.makedirs(opt.record, exist_ok=True)
        np.savez_compressed(os.path.join(opt.record, f"hertz_{variant}_L{level}.npz"), X0=Xs, T=Ts,
                            frames=np.array(frames), surf=surf, tris=tris, fz=fc[surf, 2], area=area[surf],
                            p_cons=p_cons, F=np.array([s["F"] for s in steps]))
    return dict(level=level, variant=variant, hc=hc, nodes=ns, tets=len(Ts), plane_nodes=n_plane,
                steps=steps, profile=profile, wall=round(time.time() - t0, 1))


def curved_faces(Xs, Ts):
    z_top = Xs[:, 2].max()
    faces = boundary_faces_t10(Ts)
    c = Xs[faces[:, :3]]
    flat = (np.abs(c[..., 0]) < 1e-9).all(1) | (np.abs(c[..., 1]) < 1e-9).all(1) | (np.abs(c[..., 2] - z_top) < 1e-9).all(1)
    return faces[~flat]


def post_consistent(path, opt):
    """Add the consistent pressure to an existing hertz.json without re-solving.

    The meshes are regenerated with the run's options (gmsh is deterministic here), the nodal
    forces are rebuilt as p * lumped area, and M_s p_cons = f is solved on the rest surface."""
    D = json.load(open(path))
    o = argparse.Namespace(**D["options"])
    for res in D["results"]:
        hc = o.a_ref / res["level"]
        Xs, Ts = mesh_quarter_hemisphere(hc, o.hmax, o.grad, o.gap0, o.fine_radius * o.a_ref)
        assert len(Xs) == res["nodes"], "mesh differs from the run"
        tris = sob.surface_triangles_4way(Ts)
        on = lambda v, ax, val: np.abs(Xs[v, ax] - val) < 1e-9
        z_top = Xs[:, 2].max()
        drop = np.array([all(on(t, 0, 0.0)) or all(on(t, 1, 0.0)) or all(on(t, 2, z_top)) for t in tris])
        tris = tris[~drop]
        area = sob.vertex_areas(Xs, tris, len(Xs))
        surf = np.unique(tris)
        faces = curved_faces(Xs, Ts)
        f_n = np.array(res["profile"]["p"]) * area[surf]
        assert abs(4 * f_n.sum() - res["steps"][-1]["F"]) < 1e-6 * res["steps"][-1]["F"]
        res["profile"]["p_cons"] = consistent_pressure(Xs, faces, surf, f_n).tolist()
    json.dump(D, open(path, "w"))
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="3,5,8", help="contact element size = a_ref / level")
    ap.add_argument("--fine-radius", type=float, default=1.3, help="uniform-size zone radius / a_ref")
    ap.add_argument("--variants", default="analytic,convergent,plain")
    ap.add_argument("--a-ref", type=float, default=0.012, help="target Hertz contact radius (m)")
    ap.add_argument("--indent", type=float, default=1.1e-3, help="top-face travel after first touch (m)")
    ap.add_argument("--gap0", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--dhat", type=float, default=2e-5)
    ap.add_argument("--kappa", type=float, default=1e3)
    ap.add_argument("--hmax", type=float, default=0.025)
    ap.add_argument("--grad", type=float, default=0.25)
    ap.add_argument("--plane-L", type=float, default=0.06)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--post", default="", help="add consistent pressure to an existing hertz.json and exit")
    ap.add_argument("--record", default="", help="directory for per-case .npz with node positions per load step")
    opt = ap.parse_args()
    if opt.post:
        post_consistent(opt.post, opt)
        return
    cases = [(int(l), v) for l in opt.levels.split(",") for v in opt.variants.split(",")]
    results = []
    if opt.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(opt.workers) as ex:
            for res in ex.map(run_case, *zip(*cases), [opt] * len(cases)):
                results.append(res)
                print(json.dumps({k: res[k] for k in ("level", "variant", "nodes", "tets", "wall")}), flush=True)
    else:
        for l, v in cases:
            results.append(run_case(l, v, opt))
    for res in results:
        last = res["steps"][-1]
        print(f"{res['variant']:10s} L{res['level']:<3d} nodes={res['nodes']:6d} F={last['F']:.1f} N "
              f"a_err={100 * (last['a_eff'] / last['a_hertz'] - 1):+.1f}% "
              f"p_err={100 * (last['p_max'] / last['p0_hertz'] - 1):+.1f}% "
              f"approach_err={100 * (last['approach'] / last['approach_hertz'] - 1):+.1f}% wall={res['wall']}s")
    if opt.out:
        os.makedirs(opt.out, exist_ok=True)
        json.dump(dict(options=vars(opt), results=results), open(os.path.join(opt.out, "hertz.json"), "w"))


if __name__ == "__main__":
    main()
