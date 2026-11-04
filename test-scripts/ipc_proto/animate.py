#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""
Side-by-side animation of two recorded runs (sphere_on_box.py --record). For each run: the
whole sphere in cross-section (true scale), a close-up of the contact, and below both, the
minimum gap and the sphere's vertical velocity over time. One video frame per solver step.

    uv run test-scripts/ipc_proto/animate.py --ipc ipc_1ms.npz --pen pen_1ms.npz \
        --out test-scripts/ipc_proto/out/ipc_vs_penalty_1ms.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sphere_on_box as sob  # noqa: E402

COLORS = {"ipc": "#2563eb", "penalty": "#dc2626"}
BOX = dict(color="#d1d5db", alpha=0.85, lw=0, zorder=2)


def all_faces_4way(T):
    """Every distinct tet face (interior and boundary), split into 4 linear triangles."""
    slot = {frozenset(e): 4 + k for k, e in enumerate(sob.EDGES)}
    seen = {}
    for t in T:
        for f in sob.FACES:
            seen.setdefault(tuple(sorted(t[list(f)])), (t, f))
    tris = []
    for t, f in seen.values():
        c0, c1, c2 = (t[i] for i in f)
        m01, m12, m02 = (t[slot[frozenset(p)]] for p in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])))
        tris += [(c0, m01, m02), (m01, c1, m12), (m02, m12, c2), (m01, m12, m02)]
    return np.array(tris)


def slice_tris(X, tris, y0):
    """Segments where triangles cross the plane y = y0 in the rest geometry. Each endpoint is
    stored as (i, j, s) on edge i-j, so the section follows the material as the nodes move."""
    s = X[tris, 1] - y0
    pos = s > 0
    I, J, S = [], [], []
    for tri, st in zip(tris[pos.any(1) & ~pos.all(1)], s[pos.any(1) & ~pos.all(1)]):
        ends = [(tri[a], tri[b], st[a] / (st[a] - st[b])) for a, b in ((0, 1), (1, 2), (2, 0)) if (st[a] > 0) != (st[b] > 0)]
        if len(ends) == 2:
            I.append([e[0] for e in ends])
            J.append([e[1] for e in ends])
            S.append([e[2] for e in ends])
    return np.array(I), np.array(J), np.array(S)


def chain_loop(sec):
    """Order the outline segments into one closed polygon (consecutive segments share a mesh edge)."""
    I, J, S = sec
    key = lambda k, c: (min(I[k, c], J[k, c]), max(I[k, c], J[k, c]))
    owners = {}
    for k in range(len(I)):
        for c in (0, 1):
            owners.setdefault(key(k, c), []).append((k, c))
    k, c, loop = 0, 0, []
    for _ in range(len(I)):
        loop.append((I[k, c], J[k, c], S[k, c]))
        nxt = [o for o in owners[key(k, 1 - c)] if o[0] != k][0]
        k, c = nxt
    loop = np.array(loop)
    return loop[:, 0].astype(int), loop[:, 1].astype(int), loop[:, 2]


def segments(sec, x):
    I, J, S = sec
    P = x[I] + S[..., None] * (x[J] - x[I])
    return P[..., [0, 2]] * 1e3  # mm


def load(path, label):
    d = np.load(path)
    X0, T = d["X0"], d["T"]
    y0 = X0[:, 1].mean() + 1e-7  # off any node
    gap = d["dmin"] * 1e3
    gap[0] = X0[:, 2].min() * 1e3  # box top is z = 0
    outline = slice_tris(X0, sob.surface_triangles_4way(T), y0)
    return dict(
        label=label, contact=str(d["contact"]), frames=d["frames"], t=d["t"] * 1e3, gap=gap, vz=d["com_vz"],
        dt=float(d["dt"]), xc=X0[:, 0].mean() * 1e3, zc=X0[:, 2].mean() * 1e3, R=np.ptp(X0[:, 2]) * 500,
        mesh=slice_tris(X0, all_faces_4way(T), y0), outline=outline, loop=chain_loop(outline),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ipc", required=True)
    p.add_argument("--pen", required=True)
    p.add_argument("--out", required=True, help=".mp4 or .gif")
    p.add_argument("--ipc-label", default="IPC barrier, implicit (BDF2)")
    p.add_argument("--pen-label", default="Penalty k = 1e9, explicit coupling (BE)")
    p.add_argument("--fps", type=float, default=0, help="default: about 8 s for the whole run")
    p.add_argument("--half-width", type=float, default=60.0, help="close-up half width, mm")
    p.add_argument("--zlim", type=float, nargs=2, default=(-4.5, 7.5), help="close-up z range, mm")
    p.add_argument("--gap-max", type=float, default=4.0, help="top of the gap plot, mm")
    a = p.parse_args()

    runs = [load(a.ipc, a.ipc_label), load(a.pen, a.pen_label)]
    n = max(len(r["t"]) for r in runs)
    dt = runs[0]["dt"] * 1e3
    fps = a.fps or max(1.0, round(n / 8.0))
    hold = int(round(fps * 1.5))
    order = [0] * hold + list(range(n)) + [n - 1] * 2 * hold

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.8, 10.8), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.1, 1, 1.05], hspace=0.42, wspace=0.16, left=0.06, right=0.98, top=0.9,
                          bottom=0.05)
    full_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    zoom_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axg, axv = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    fig.suptitle(f"T10 sphere (1 kg, R = 150 mm) dropped onto a rigid box at 1 m/s, dt = {dt:g} ms", fontsize=14, y=0.985)
    clock = fig.text(0.5, 0.945, "", ha="center", fontsize=12, color="#374151")

    artists = []
    for axf, axz, r in zip(full_axes, zoom_axes, runs):
        col = COLORS[r["contact"]]
        R, xc = r["R"], r["xc"]
        # whole sphere, true scale
        axf.set_aspect("equal")
        axf.set_xlim(xc - 1.45 * R, xc + 1.45 * R)
        axf.set_ylim(-0.3 * R, 2.35 * R)
        axf.add_patch(Rectangle((xc - 3 * R, -R), 6 * R, R, **BOX))
        axf.axhline(0, color="#4b5563", lw=1.2, zorder=2)
        axf.text(xc - 1.4 * R, -0.22 * R, "rigid box", color="#4b5563", fontsize=9, zorder=4)
        axf.set_title(r["label"], color=col, fontsize=13, fontweight="bold")
        axf.set_xlabel("x (mm)")
        axf.set_ylabel("z (mm)")
        axf.add_patch(Rectangle((xc - a.half_width, a.zlim[0]), 2 * a.half_width, a.zlim[1] - a.zlim[0], fill=False,
                                ec="#111827", lw=0.8, ls="--", zorder=6))
        axf.text(xc + a.half_width + 4, a.zlim[1] + 4, "close-up below", fontsize=8.5, color="#374151", zorder=6)
        # close-up of the contact, stretched vertically
        axz.set_xlim(xc - a.half_width, xc + a.half_width)
        axz.set_ylim(*a.zlim)
        axz.add_patch(Rectangle((xc - 2 * a.half_width, a.zlim[0] - 10), 4 * a.half_width, 10 - a.zlim[0], **BOX))
        axz.axhline(0, color="#4b5563", lw=1.2, zorder=2)
        axz.set_xlabel("x (mm)")
        axz.set_ylabel("z (mm)")
        per_ax = []
        for ax in (axf, axz):
            fill = Polygon(np.zeros((3, 2)), closed=True, fc=col, alpha=0.12, lw=0, zorder=1)
            ax.add_patch(fill)
            mesh = LineCollection([], colors="#6b7280", linewidths=0.35, alpha=0.6, zorder=1)
            outline = LineCollection([], colors=col, linewidths=1.8, zorder=3)
            ax.add_collection(mesh)
            ax.add_collection(outline)
            per_ax.append((fill, mesh, outline))
        (low,) = axz.plot([], [], "o", ms=5, color=col, zorder=5)
        note = axz.text(0.02, 0.93, "", transform=axz.transAxes, va="top", fontsize=11, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e5e7eb"))
        artists.append((per_ax, low, note))

    fig.canvas.draw()
    for ax in zoom_axes:
        bb = ax.get_window_extent()
        ex = (2 * a.half_width / bb.width) / ((a.zlim[1] - a.zlim[0]) / bb.height)
        ax.text(0.98, 0.93, f"contact close-up, vertical scale x{ex:.0f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="#6b7280")

    tmax = max(r["t"][-1] for r in runs)
    gmin = min(r["gap"].min() for r in runs)
    traces = []
    for ax, key, title, unit in ((axg, "gap", "minimum gap: sphere surface to box", "mm"), (axv, "vz", "sphere vertical velocity", "m/s")):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(unit)
        ax.set_xlim(0, tmax)
        ax.axhline(0, color="#4b5563", lw=0.8)
        for r in runs:
            ax.plot(r["t"], r[key], color=COLORS[r["contact"]], alpha=0.18, lw=1.5)
        lines = [ax.plot([], [], color=COLORS[r["contact"]], lw=2, marker="o", ms=3, label=r["label"])[0] for r in runs]
        dots = [ax.plot([], [], "o", ms=7, color=COLORS[r["contact"]])[0] for r in runs]
        cursor = ax.axvline(0, color="#9ca3af", lw=1, ls="--")
        traces.append((key, lines, dots, cursor))
    gtop = min(a.gap_max, max(r["gap"].max() for r in runs) * 1.1)
    axg.set_ylim(min(-0.5, 1.15 * gmin), gtop)
    if gtop < max(r["gap"].max() for r in runs):
        axg.text(0.99, 0.96, f"zoomed to {gtop:g} mm: the gap after the bounce runs off the top",
                 transform=axg.transAxes, ha="right", va="top", fontsize=8.5, color="#6b7280")
    axg.axhspan(axg.get_ylim()[0], 0, color="#dc2626", alpha=0.06, lw=0)
    axg.text(0.99, 0.04, "below 0 = penetration", transform=axg.transAxes, ha="right", fontsize=8.5, color="#b91c1c")
    axv.legend(loc="lower right", fontsize=9, frameon=False)

    def draw(i):
        for r, (per_ax, low, note) in zip(runs, artists):
            k = min(i, len(r["t"]) - 1)
            x = r["frames"][k]
            poly = segments(r["loop"], x)
            mesh = segments(r["mesh"], x)
            seg = segments(r["outline"], x)
            for fill, ml, ol in per_ax:
                fill.set_xy(poly)
                ml.set_segments(mesh)
                ol.set_segments(seg)
            j = poly[:, 1].argmin()
            low.set_data([poly[j, 0]], [poly[j, 1]])
            g = r["gap"][k]
            if g < 0:
                note.set_text(f"penetration {-g:.2f} mm")
                note.set_color("#b91c1c")
            elif g < 1.0:
                note.set_text(f"gap {g * 1e3:.0f} µm")
                note.set_color("#111827")
            else:
                note.set_text(f"gap {g:.2f} mm")
                note.set_color("#111827")
        for key, lines, dots, cursor in traces:
            for r, ln, dot in zip(runs, lines, dots):
                k = min(i, len(r["t"]) - 1)
                ln.set_data(r["t"][: k + 1], r[key][: k + 1])
                dot.set_data([r["t"][k]], [r[key][k]])
            cursor.set_xdata([runs[0]["t"][min(i, len(runs[0]["t"]) - 1)]] * 2)
        clock.set_text(f"t = {runs[0]['t'][min(i, len(runs[0]['t']) - 1)]:.1f} ms   (step {i} of {n - 1})")
        return []

    anim = FuncAnimation(fig, draw, frames=order, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = PillowWriter(fps=fps) if a.out.endswith(".gif") else FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(a.out, writer=writer)
    draw(min(n - 1, int(np.argmin(runs[1]["gap"]))))  # still: deepest penalty penetration
    fig.savefig(os.path.splitext(a.out)[0] + "_still.png")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
