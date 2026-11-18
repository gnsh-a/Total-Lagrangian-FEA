#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""
One brick-on-slope case on two slopes: IPC prototype (brick_slope.py recording) vs stock TL-FEA + DEME
(lib_bin/ptest/brick_sliding .vtu frames + rigid_equiv.csv), one row per slope, live graphs of the slid
distance and the gap / penetration. Both use sudden release from rest.

    uv run test-scripts/ipc_proto/animate_case.py --title "Case 1: frictionless" --tag _mu0 --mu 0.0 \
        --row 01=~/work/fea/ref-runs/2fffca8/output/brick_sliding_mu0_s01_ctype0_E1e7 \
        --row 04=~/work/fea/ref-runs/2fffca8/output/brick_sliding_mu0_s04_ctype0_E1e7 \
        --out test-scripts/ipc_proto/out/brick_slope/case1_frictionless.mp4

--tag / --mu pick the IPC files ipc{tag}_s{slope}_mu{mu}.json and rec{tag}_s{slope}_mu{mu}.npz;
--mu is mu_k, --mu-s > 0 gives the static coefficient for the analytic answer.
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
from animate_brick import COL, read_vtu_points  # noqa: E402
from brick_slope import ALPHA, BRICK, SLOPE, analytic  # noqa: E402

DT = 5e-4


def load_row(key, deme_dir, a, Xb):
    """Brick frames, time series and analytic answer for one slope."""
    alpha = ALPHA[key]
    n = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])  # slope normal (up)
    up = np.array([np.cos(alpha), 0.0, np.sin(alpha)])  # along the slope, uphill
    nb = len(Xb)

    # IPC: every step
    ipc_frames = np.load(os.path.join(a.ipc, f"rec{a.tag}_s{key}_mu{a.mu}.npz"))["frames"]
    ipc = json.load(open(os.path.join(a.ipc, f"ipc{a.tag}_s{key}_mu{a.mu}.json")))

    # DEME: .vtu every few steps (step k is written after its solve, t = (k+1) dt), csv every step
    d = os.path.expanduser(deme_dir)
    vtus = sorted(glob.glob(os.path.join(d, "mesh_*.vtu")), key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[0]))
    steps = np.array([int(re.findall(r"\d+", os.path.basename(f))[0]) for f in vtus])
    deme_frames = np.array([read_vtu_points(f)[-nb:] for f in vtus])  # brick is loaded after the slope
    csv = np.genfromtxt(os.path.join(d, "rigid_equiv.csv"), delimiter=",", names=True, dtype=None, encoding="utf-8")
    csv = csv[csv["body"] == "brick"]
    com_d = np.column_stack([csv["com_x"], csv["com_y"], csv["com_z"]])

    # slope plane (as brick_sliding.cc: highest slope node under the brick footprint)
    Xs, Ts = sob.read_tetgen_t10(SLOPE.format(key))
    lo, hi = deme_frames[0][:, :2].min(0) - 0.02, deme_frames[0][:, :2].max(0) + 0.02
    b = float((Xs[((Xs[:, :2] >= lo) & (Xs[:, :2] <= hi)).all(1)] @ n).max())

    g_i = np.r_[np.nan, ipc["dmin"]] * 1e6
    g_i[0] = (ipc_frames[0] @ n - b).min() * 1e6
    tt = np.linspace(0, 0.3, 301)
    s_ref = analytic(alpha, float(a.mu), tt, a.mu_s)[0] * 1e3
    s_end = (com_d[-1] - com_d[0]) @ -up * 1e3
    return dict(
        key=key, alpha=alpha, steps=steps, ipc_frames=ipc_frames, deme_frames=deme_frames,
        slope_xz=anim.segments(anim.chain_loop(anim.slice_tris(Xs, sob.surface_triangles_4way(Ts), 1e-7)), Xs),
        t_i=np.array(ipc["t"]), s_i=np.array(ipc["s"]) * 1e3, g_i=g_i,
        t_d=csv["time"], s_d=(com_d - com_d[0]) @ -up * 1e3,
        t_df=(steps + 1) * DT, g_d=np.array([(f @ n - b).min() for f in deme_frames]) * 1e6,
        tt=tt, s_ref=s_ref, s_end=s_end)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--title", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--mu", default="0.2", help="mu_k, as written in the IPC file names")
    p.add_argument("--mu-s", type=float, default=-1.0)
    p.add_argument("--row", action="append", required=True, help="SLOPE=DEME_DIR, one per row")
    p.add_argument("--ipc", default=os.path.join(HERE, "out", "brick_slope"))
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=12)
    a = p.parse_args()

    Xb, Tb = sob.read_tetgen_t10(BRICK)
    b_mesh = anim.slice_tris(Xb, anim.all_faces_4way(Tb), 1e-7)
    b_loop = anim.chain_loop(anim.slice_tris(Xb, sob.surface_triangles_4way(Tb), 1e-7))
    rows = [load_row(*r.split("=", 1), a, Xb) for r in a.row]
    nf = min(len(r["steps"]) for r in rows)
    steps = rows[0]["steps"][:nf]

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[1.3, 1.3, 1, 1], hspace=0.42, wspace=0.3, left=0.1,
                          right=0.975, top=0.84, bottom=0.07)
    mus = f"μs = {a.mu_s}, μk = {a.mu}" if a.mu_s > 0 else f"μ = {a.mu}"
    fig.suptitle(f"{a.title} ({mus} in both)", fontsize=17, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925, "1 kg T10 brick released at rest, dt = 0.5 ms.   "
             "IPC prototype: implicit barrier + smoothed Coulomb.   Stock DEME: explicit Hertz + spring friction.",
             ha="center", fontsize=11.5, color="#374151")
    clock = fig.text(0.5, 0.885, "", ha="center", fontsize=13, color="#111827")

    art, live = [], []
    for i, r in enumerate(rows):
        will_slide = r["s_ref"][-1] > 0
        verdict = f"analytic: slides {r['s_ref'][-1]:.1f} mm in 0.3 s" if will_slide else "analytic: sticks"
        pos = gs[i, 0].get_position(fig)
        fig.text(0.03, (pos.y0 + pos.y1) / 2, f"slope {r['key']}: α = {np.degrees(r['alpha']):.2f}°\ntan α = {np.tan(r['alpha']):.3f}\n{verdict}",
                 rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")

        # world-frame views
        allx = np.concatenate([r["ipc_frames"][:, :, [0, 2]].reshape(-1, 2),
                               r["deme_frames"][:nf, :, [0, 2]].reshape(-1, 2)]) * 1e3
        xmin, zmin = allx.min(0) - 25
        xmax, zmax = allx.max(0) + 25
        views = []
        for j, (name, col) in enumerate([("IPC prototype", COL["ipc"]), ("Stock DEME (TL-FEA)", COL["deme"])]):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect("equal")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(zmin, zmax)
            ax.add_patch(Polygon(r["slope_xz"], closed=True, fc="#d1d5db", ec="#4b5563", lw=1, zorder=2))
            ax.set_title(name, color=col, fontsize=13, fontweight="bold")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
            bf = Polygon(np.zeros((3, 2)), closed=True, fc=col, alpha=0.15, lw=0, zorder=3)
            bm = LineCollection([], colors="#6b7280", linewidths=0.3, alpha=0.6, zorder=3)
            bo = Polygon(np.zeros((3, 2)), closed=True, fill=False, ec=col, lw=1.6, zorder=4)
            ax.add_patch(bf)
            ax.add_collection(bm)
            ax.add_patch(bo)
            note = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top", fontsize=12, zorder=7,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb"))
            views.append((bf, bm, bo, note))
        art.append(views)

        # live graphs
        series = [("distance slid down the slope (mm)", (r["t_i"], r["s_i"]), (r["t_d"], r["s_d"]), (r["tt"], r["s_ref"])),
                  ("gap (+) / penetration (−), µm", (r["t_i"], r["g_i"]), (r["t_df"], r["g_d"]), None)]
        for j, (title, si, sd, sr) in enumerate(series):
            ax = fig.add_subplot(gs[i, 2 + j])
            ax.set_title(title, fontsize=11.5)
            ax.set_xlabel("t (s)")
            ax.set_xlim(0, 0.3)
            ax.axhline(0, color="#4b5563", lw=0.7)
            ax.plot(*si, color=COL["ipc"], alpha=0.15, lw=1.5)
            ax.plot(*sd, color=COL["deme"], alpha=0.15, lw=1.5)
            if sr is not None:
                ax.plot(*sr, color=COL["ref"], lw=1.4, ls="--", label="analytic", zorder=5)
            li, = ax.plot([], [], color=COL["ipc"], lw=2.2, label="IPC")
            ld, = ax.plot([], [], color=COL["deme"], lw=2.2, label="DEME")
            cur = ax.axvline(0, color="#9ca3af", lw=1, ls="--")
            vals = np.concatenate([si[1][np.isfinite(si[1])], sd[1][np.isfinite(sd[1])]] + ([sr[1]] if sr else []))
            pad = 0.08 * (vals.max() - vals.min() + 1e-9)
            ax.set_ylim(vals.min() - pad, vals.max() + pad)
            if j == 0:
                ax.legend(fontsize=10, frameon=False, loc="upper left")
            live.append((li, ld, cur, si, sd))

    def draw(k):
        tk = (steps[k] + 1) * DT
        for r, views in zip(rows, art):
            ki = min(steps[k] + 1, len(r["ipc_frames"]) - 1)
            kd = int(np.searchsorted(r["t_d"], tk - 1e-9))
            for (bf, bm, bo, note), x, s, g in [
                    (views[0], r["ipc_frames"][ki], r["s_i"][min(ki, len(r["s_i"]) - 1)], r["g_i"][ki]),
                    (views[1], r["deme_frames"][k], r["s_d"][min(kd, len(r["s_d"]) - 1)], r["g_d"][k])]:
                loop = anim.segments(b_loop, x)
                bf.set_xy(loop)
                bo.set_xy(loop)
                bm.set_segments(anim.segments(b_mesh, x))
                note.set_text(f"slid {s:.2f} mm\n" + (f"gap {g:.0f} µm" if g >= 0 else f"penetration {-g:.0f} µm"))
                note.set_color("#111827" if g >= 0 else "#b91c1c")
        for li, ld, cur, si, sd in live:
            m = si[0] <= tk + 1e-12
            li.set_data(si[0][m], si[1][m])
            m = sd[0] <= tk + 1e-12
            ld.set_data(sd[0][m], sd[1][m])
            cur.set_xdata([tk, tk])
        clock.set_text(f"t = {tk:.3f} s   (step {steps[k] + 1} of 600)")
        return []

    hold = int(a.fps)
    order = [0] * hold + list(range(nf)) + [nf - 1] * 2 * hold
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    FuncAnimation(fig, draw, frames=order, blit=False).save(a.out, writer=FFMpegWriter(fps=a.fps, bitrate=4000))
    draw(nf - 1)
    fig.savefig(os.path.splitext(a.out)[0] + "_still.png")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
