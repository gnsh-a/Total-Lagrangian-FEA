#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""
Side-by-side animation of the brick-on-slope test: IPC prototype (brick_slope.py recording)
vs TL-FEA + DEME (lib_bin/ptest/brick_sliding .vtu frames + rigid_equiv.csv), same mu in both.

    uv run test-scripts/ipc_proto/animate_brick.py --slope 04 --mu 0.2 \
        --deme ~/work/fea/ref-runs/2fffca8/output/brick_sliding_mu0.2_s04_ctype0_E1e7 \
        --out test-scripts/ipc_proto/out/brick_slope/ipc_vs_deme_s04.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sphere_on_box as sob  # noqa: E402
import animate as anim  # noqa: E402
from brick_slope import ALPHA, BRICK, SLOPE, G  # noqa: E402

COL = {"ipc": "#2563eb", "deme": "#dc2626", "ref": "#111827"}


def read_vtu_points(path):
    txt = open(path).read()
    block = re.search(r"<Points>\s*<DataArray[^>]*>(.*?)</DataArray>", txt, re.S).group(1)
    return np.array(block.split(), float).reshape(-1, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slope", required=True)
    p.add_argument("--mu", type=float, default=0.2)
    p.add_argument("--ipc", default=os.path.join(HERE, "out", "brick_slope"))
    p.add_argument("--deme", required=True, help="brick_sliding output folder (.vtu frames + rigid_equiv.csv)")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=12)
    a = p.parse_args()

    key, mu, dt = a.slope, a.mu, 5e-4
    alpha = ALPHA[key]
    n = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])  # slope normal (up)
    up = np.array([np.cos(alpha), 0.0, np.sin(alpha)])  # along the slope, uphill
    Xb, Tb = sob.read_tetgen_t10(BRICK)
    nb = len(Xb)

    # IPC: every step
    rec = np.load(os.path.join(a.ipc, f"rec_s{key}_mu{mu}.npz"))
    ipc_frames = rec["frames"]
    ipc = json.load(open(os.path.join(a.ipc, f"ipc_s{key}_mu{mu}.json")))
    t_ipc = np.array(ipc["t"])

    # DEME: .vtu every few steps (step k is written after its solve, t = (k+1) dt), csv every step
    vtus = sorted(glob.glob(os.path.join(a.deme, "mesh_*.vtu")), key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[0]))
    steps = np.array([int(re.findall(r"\d+", os.path.basename(f))[0]) for f in vtus])
    deme_frames = np.array([read_vtu_points(f)[-nb:] for f in vtus])  # brick is loaded after the slope
    csv = np.genfromtxt(os.path.join(a.deme, "rigid_equiv.csv"), delimiter=",", names=True, dtype=None, encoding="utf-8")
    csv = csv[csv["body"] == "brick"]
    t_deme = csv["time"]
    com_d = np.column_stack([csv["com_x"], csv["com_y"], csv["com_z"]])
    v_d = np.column_stack([csv["vcom_x"], csv["vcom_y"], csv["vcom_z"]])

    # slope plane (as brick_sliding.cc: highest slope node under the brick footprint)
    Xs, Ts = sob.read_tetgen_t10(SLOPE.format(key))
    lo, hi = deme_frames[0][:, :2].min(0) - 0.02, deme_frames[0][:, :2].max(0) + 0.02
    b = float((Xs[((Xs[:, :2] >= lo) & (Xs[:, :2] <= hi)).all(1)] @ n).max())

    # cross-sections at y = 0 (brick: mesh lines + outline loop; slope: outline loop)
    y0 = 1e-7
    b_mesh = anim.slice_tris(Xb, anim.all_faces_4way(Tb), y0)
    b_loop = anim.chain_loop(anim.slice_tris(Xb, sob.surface_triangles_4way(Tb), y0))
    s_loop = anim.chain_loop(anim.slice_tris(Xs, sob.surface_triangles_4way(Ts), y0))
    slope_xz = anim.segments(s_loop, Xs)  # mm
    to_world = lambda sec, x: anim.segments(sec, x)  # (x, z) in mm

    def to_slope(sec, x):
        """(along slope from the start centre in mm, normal offset from the slope surface in um)"""
        I, J, S = sec
        P = x[I] + S[..., None] * (x[J] - x[I])
        return np.stack([(P @ up - c0) * 1e3, (P @ n - b) * 1e6], axis=-1)

    c0 = float(ipc_frames[0].mean(0) @ up)

    # time series: slid distance (downhill), speed, normal position of the lowest brick point
    s_i, v_i = np.array(ipc["s"]) * 1e3, np.array(ipc["v"]) * 1e3
    g_i = np.r_[np.nan, ipc["dmin"]] * 1e6
    g_i[0] = (ipc_frames[0] @ n - b).min() * 1e6
    s_d = (com_d - com_d[0]) @ -up * 1e3
    v_d = v_d @ -up * 1e3
    t_df = (steps + 1) * dt
    g_d = np.array([(f @ n - b).min() for f in deme_frames]) * 1e6
    acc = G * (np.sin(alpha) - mu * np.cos(alpha))
    tt = np.linspace(0, 0.3, 301)
    s_ref = np.where(acc > 0, 0.5 * acc * tt**2, 0.0) * 1e3
    v_ref = np.where(acc > 0, acc * tt, 0.0) * 1e3
    verdict = (f"should slide: a = g(sin a - mu cos a) = {acc:.3f} m/s^2" if acc > 0
               else "should stick: tan a < mu")

    runs = [
        dict(name="IPC prototype: implicit barrier + smoothed Coulomb", col=COL["ipc"],
             frame=lambda k: ipc_frames[min(steps[k] + 1, len(ipc_frames) - 1)],
             gap=lambda k: g_i[min(steps[k] + 1, len(g_i) - 1)]),
        dict(name="TL-FEA + DEME: explicit Hertz + spring friction", col=COL["deme"],
             frame=lambda k: deme_frames[k], gap=lambda k: g_d[k]),
    ]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.8, 10.0), dpi=100)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.35, 1.0, 1.0], hspace=0.45, wspace=0.6, left=0.06, right=0.98,
                          top=0.89, bottom=0.06)
    full = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
    zoom = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    graphs = [fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]), fig.add_subplot(gs[2, 4:6])]
    fig.suptitle(f"1 kg T10 brick released on slope {key} (tan a = {np.tan(alpha):.3f}), mu = {mu} in both "
                 f"-- {verdict}", fontsize=12.5, y=0.985)
    clock = fig.text(0.5, 0.935, "", ha="center", fontsize=12, color="#374151")

    allx = np.concatenate([ipc_frames[:, :, [0, 2]].reshape(-1, 2), deme_frames[:, :, [0, 2]].reshape(-1, 2)]) * 1e3
    xmin, zmin = allx.min(0) - 25
    xmax, zmax = allx.max(0) + 25
    zl = (min(-450.0, np.nanmin(g_d) * 1.25), max(150.0, np.nanmax(g_i[1:]) * 3))
    art = []
    for axf, axz, r in zip(full, zoom, runs):
        axf.set_aspect("equal")
        axf.set_xlim(xmin, xmax)
        axf.set_ylim(zmin, zmax)
        axf.add_patch(Polygon(slope_xz, closed=True, fc="#d1d5db", ec="#4b5563", lw=1, zorder=2))
        axf.set_title(r["name"], color=r["col"], fontsize=11.5, fontweight="bold")
        axf.set_xlabel("x (mm)")
        axf.set_ylabel("z (mm)")
        bf = Polygon(np.zeros((3, 2)), closed=True, fc=r["col"], alpha=0.15, lw=0, zorder=3)
        bm = LineCollection([], colors="#6b7280", linewidths=0.3, alpha=0.6, zorder=3)
        bo = Polygon(np.zeros((3, 2)), closed=True, fill=False, ec=r["col"], lw=1.6, zorder=4)
        axf.add_patch(bf)
        axf.add_collection(bm)
        axf.add_patch(bo)
        # close-up in the slope frame: along-slope (mm) vs normal offset from the surface (um)
        axz.set_xlim(-140, 130)
        axz.set_ylim(*zl)
        axz.axhspan(zl[0] - 1e4, 0, color="#d1d5db", alpha=0.85, lw=0, zorder=2)
        axz.axhline(0, color="#4b5563", lw=1.2, zorder=2)
        axz.text(-136, zl[0] + 0.06 * (zl[1] - zl[0]), "slope surface (below = inside the slope)", fontsize=8.5,
                 color="#4b5563", zorder=5)
        axz.set_xlabel("along slope, uphill + (mm)")
        axz.set_ylabel("normal to slope (um)")
        zf = Polygon(np.zeros((3, 2)), closed=True, fc=r["col"], alpha=0.15, lw=0, zorder=3)
        zo = Polygon(np.zeros((3, 2)), closed=True, fill=False, ec=r["col"], lw=1.8, zorder=4)
        axz.add_patch(zf)
        axz.add_patch(zo)
        note = axz.text(0.02, 0.92, "", transform=axz.transAxes, va="top", fontsize=11, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb"))
        axz.set_title("close-up of the brick bottom (normal axis in um, along-slope in mm)", fontsize=9.5,
                      color="#374151")
        art.append((bf, bm, bo, zf, zo, note))

    series = [
        ("distance slid down the slope", "mm", (t_ipc, s_i), (csv["time"], s_d), (tt, s_ref)),
        ("speed down the slope", "mm/s", (t_ipc, v_i), (csv["time"], v_d), (tt, v_ref)),
        ("lowest brick point vs slope surface", "um", (t_ipc, g_i), (t_df, g_d), None),
    ]
    live = []
    for ax, (title, unit, si, sd, sr) in zip(graphs, series):
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(unit)
        ax.set_xlim(0, 0.3)
        ax.axhline(0, color="#4b5563", lw=0.7)
        ax.plot(*si, color=COL["ipc"], alpha=0.15, lw=1.5)
        ax.plot(*sd, color=COL["deme"], alpha=0.15, lw=1.5)
        if sr is not None:
            ax.plot(*sr, color=COL["ref"], lw=1, ls="--", label="analytic")
        li, = ax.plot([], [], color=COL["ipc"], lw=2, label="IPC")
        ld, = ax.plot([], [], color=COL["deme"], lw=2, label="DEME")
        cur = ax.axvline(0, color="#9ca3af", lw=1, ls="--")
        live.append((li, ld, cur, si, sd))
    graphs[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    graphs[2].text(0.98, 0.55, "+ gap / - penetration", transform=graphs[2].transAxes, ha="right",
                   fontsize=8.5, color="#6b7280")

    def draw(k):
        tk = (steps[k] + 1) * dt
        for r, (bf, bm, bo, zf, zo, note) in zip(runs, art):
            x = r["frame"](k)
            loop = to_world(b_loop, x)
            bf.set_xy(loop)
            bo.set_xy(loop)
            bm.set_segments(to_world(b_mesh, x))
            zl_ = to_slope(b_loop, x)
            zf.set_xy(zl_)
            zo.set_xy(zl_)
            g = r["gap"](k)
            note.set_text(f"gap {g:.0f} um" if g >= 0 else f"penetration {-g:.0f} um")
            note.set_color("#111827" if g >= 0 else "#b91c1c")
        for li, ld, cur, si, sd in live:
            m = si[0] <= tk + 1e-12
            li.set_data(si[0][m], si[1][m])
            m = sd[0] <= tk + 1e-12
            ld.set_data(sd[0][m], sd[1][m])
            cur.set_xdata([tk, tk])
        clock.set_text(f"t = {tk:.3f} s   (step {steps[k] + 1} of 600, dt = 0.5 ms)")
        return []

    for ax, (title, unit, si, sd, sr) in zip(graphs, series):
        vals = np.concatenate([si[1][np.isfinite(si[1])], sd[1][np.isfinite(sd[1])]] + ([sr[1]] if sr else []))
        pad = 0.08 * (vals.max() - vals.min() + 1e-9)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)

    hold = int(a.fps)
    order = [0] * hold + list(range(len(steps))) + [len(steps) - 1] * 2 * hold
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    FuncAnimation(fig, draw, frames=order, blit=False).save(a.out, writer=FFMpegWriter(fps=a.fps, bitrate=3000))
    draw(len(steps) - 1)
    fig.savefig(os.path.splitext(a.out)[0] + "_still.png")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
