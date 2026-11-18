#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""
Final brick-on-slope table: analytic vs IPC vs DEME vs Chrono SMC, slid distance at 0.3 s, for the cases all
three codes have (single mu: 0.2 on 8 slopes, 0.4 and 0 on slopes 01 and 04). Same scoring as
ipc_proto/analyze_brick.py: "slid" = more than 5 mm or still moving faster than 10 mm/s at 0.3 s.

    python3 test-scripts/chrono_smc/final_table.py   # writes out_final/final_table.md
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ipc_proto"))
from analyze_brick import ALPHA, analytic, load_deme, load_ipc  # noqa: E402

DEME = os.path.expanduser("~/work/fea/ref-runs/2fffca8/output/brick_sliding_mu{mu}_s{{key}}_ctype0_E1e7")
CASES = [("0.2", k) for k in ("01", "02", "03", "03p1", "03p2", "03p3", "03p4", "04")] + \
        [("0.4", "01"), ("0.4", "04"), ("0", "01"), ("0", "04")]
IPC_TAG = {"0.2": ("", "0.2"), "0.4": ("", "0.4"), "0": ("_mu0", "0.0")}


def load_smc(key, mu):
    r = np.genfromtxt(os.path.join(HERE, "out_final", f"s{key}_mu{mu}.csv"), delimiter=",", names=True)
    return r["t"], r["s"], r["vx"], float(r["z"][-1]) * 1e6


def slid(s, v):
    return s[-1] > 5e-3 or v[-1] > 10e-3


def main():
    codes = ["IPC", "DEME", "SMC"]
    rows, stats = [], {c: dict(right=0, n=0, err=[], creep=[], normal=[]) for c in codes}
    for mu, key in CASES:
        a, m = ALPHA[key], float(mu)
        got = {}
        tag, mus = IPC_TAG[mu]
        r = load_ipc(tag, key, mus)
        if r is not None:
            got["IPC"] = (r[0], r[1], r[2], r[3], r[5])
        r = load_deme(DEME.format(mu=mu), key)
        if r is not None:
            got["DEME"] = (r[0], r[1], r[2], r[3], "0")
        got["SMC"] = (*load_smc(key, mu), "0")
        t_ref = got["SMC"][0]
        s_ref = analytic(a, m, m, "0", t_ref)[0][-1]
        should = s_ref > 0
        cells = []
        for c in codes:
            if c not in got:
                cells.append("—")
                continue
            t, s, v, nrm, loading = got[c]
            ref = analytic(a, m, m, loading, t)[0][-1]
            ok = slid(s, v) == (ref > 0)
            st = stats[c]
            st["n"] += 1
            st["right"] += ok
            st["normal"].append(nrm)
            if ref > 0:
                e = (s[-1] - ref) / ref
                st["err"].append(e)
                cells.append(f"{s[-1] * 1e3:.2f} ({e * 100:+.1f}%)" + ("" if ok else " ✗"))
            else:
                st["creep"].append(s[-1])
                cells.append(f"{s[-1] * 1e3:.2f}" + ("" if ok else " ✗"))
        verdict = "slide" if should else "stick"
        rows.append(f"| {mu} | {key} | {np.tan(a):.4f} | {verdict} | {s_ref * 1e3:.2f} | " + " | ".join(cells) + " |")

    out = ["# Brick on a slope: analytic vs IPC vs DEME vs Chrono SMC", "",
           "Slid distance at 0.3 s (mm); error vs analytic for sliding cases; ✗ = wrong stick/slide outcome.", "",
           "| μ | slope | tan α | analytic | analytic s | IPC | DEME | Chrono SMC |",
           "|---|---|---|---|---|---|---|---|", *rows, "",
           "| | IPC | DEME | Chrono SMC |", "|---|---|---|---|"]
    def f(c, k):
        st = stats[c]
        e = np.abs(st["err"]) * 100 if st["err"] else np.array([np.nan])
        return {"right": f"{st['right']}/{st['n']}", "med": f"{np.median(e):.1f}%", "max": f"{np.max(e):.1f}%",
                "creep": f"{max(st['creep']) * 1e3:.2f} mm" if st["creep"] else "—",
                "normal": f"{np.mean(st['normal']):+.0f} µm"}[k]
    for lab, k in [("stick/slide outcome right", "right"), ("median sliding error", "med"),
                   ("max sliding error", "max"), ("max creep on stick cases", "creep"),
                   ("normal offset (+ gap, − penetration)", "normal")]:
        out.append(f"| {lab} | " + " | ".join(f(c, k) for c in codes) + " |")
    out += ["", "IPC and DEME: deformable T10 brick (E 1e7). Chrono SMC: rigid box brick, Chrono::Multicore SMC with",
            "contact history (MultiStep), Hertz, dt 5e-4. All single μ (μ_s = μ_k), gravity on from t = 0."]
    text = "\n".join(out)
    open(os.path.join(HERE, "out_final", "final_table.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
