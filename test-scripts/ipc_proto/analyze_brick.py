#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Score brick-on-slope runs (IPC jsons from brick_slope.py, DEME rigid_equiv.csv from brick_sliding)
against the rigid-block Coulomb answer for each run's own friction law and gravity ramp.

    uv run test-scripts/ipc_proto/analyze_brick.py \
        --ipc "eps1e-3=:0.2:-1" --ipc "sk=_sk:0.2:0.25" \
        --deme "v0=<dir>/output/brick_sliding_v0_s{key}_ctype0_E1e7:0.2:0.2:0"

Spec formats: IPC  label=tag:mu_k:mu_s (loading read from the json);\nDEME label=dir_pattern:mu_k:mu_s:loading, loading = 0 | g<T> (gravity ramp) | t<T> (along-slope ramp)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
ALPHA = {"01": 0.18, "02": 0.197395560, "03p3": float(np.arctan(0.247)), "03p4": float(np.arctan(0.248)),
         "03p1": 0.2440372654, "03": 0.244978663, "03p2": 0.2459196179, "04": 0.25}
T_END = 0.3


def analytic(alpha, mu_k, mu_s, loading, t):
    """loading: "0" sudden, "g<T>" gravity ramped over T s, "t<T>" along-slope load ramped (tilt test)."""
    sys.path.insert(0, HERE)
    from brick_slope import analytic as ref

    g = float(loading[1:]) if loading.startswith("g") else (float(loading) if loading[0].isdigit() else 0.0)
    tr = float(loading[1:]) if loading.startswith("t") else 0.0
    return ref(alpha, mu_k, t, mu_s, g, tr)


def load_ipc(tag, key, mu_k):
    path = os.path.join(HERE, "out", "brick_slope", f"ipc{tag}_s{key}_mu{mu_k}.json")
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    loading = f"t{r['tangent_ramp']}" if r.get("tangent_ramp", 0) > 0 else f"g{r.get('gravity_ramp', 0.0)}"
    return np.array(r["t"]), np.array(r["s"]), np.array(r["v"]), float(np.nanmin(r["dmin"])) * 1e6, \
        float(np.mean(r["newton"])), loading


def load_deme(pattern, key):
    path = os.path.join(os.path.expanduser(pattern.format(key=key)), "rigid_equiv.csv")
    if not os.path.exists(path):
        return None
    r = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8", invalid_raise=False)
    r = r[r["body"] == "brick"]
    a = ALPHA[key]
    up = np.array([np.cos(a), 0, np.sin(a)])
    n = np.array([-np.sin(a), 0, np.cos(a)])
    com = np.column_stack([r["com_x"], r["com_y"], r["com_z"]])
    v = np.column_stack([r["vcom_x"], r["vcom_y"], r["vcom_z"]])
    t = r["time"]
    return t, (com - com[0]) @ -up, v @ -up, float(((com - com[0]) @ n)[-1]) * 1e6, np.nan, None


def score(label, rows):
    """rows: (key, t, s, v, s_ref, v_ref, normal_um, newton)"""
    wrong, errs, creep = [], [], []
    for key, t, s, v, s_ref, v_ref, nrm, nw in rows:
        should_slide = s_ref[-1] > 0
        slid = s[-1] > 5e-3 or v[-1] > 10e-3  # > 5 mm or still moving > 10 mm/s at 0.3 s
        if slid != should_slide:
            wrong.append(key)
        if should_slide:
            errs.append((s[-1] - s_ref[-1]) / s_ref[-1])
        else:
            creep.append(s[-1])
    e = np.array(errs) if errs else np.array([np.nan])
    return dict(label=label, wrong=wrong, mean_abs_err=float(np.nanmean(np.abs(e))), max_abs_err=float(np.nanmax(np.abs(e))),
                creep_mm=float(np.max(creep) * 1e3) if creep else np.nan,
                normal_um=float(np.mean([r[6] for r in rows])),
                newton=float(np.mean([r[7] for r in rows])) if np.isfinite([r[7] for r in rows]).all() else np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ipc", action="append", default=[])
    p.add_argument("--deme", action="append", default=[])
    p.add_argument("--per-slope", action="store_true")
    p.add_argument("--slopes", default=",".join(ALPHA), help="comma-separated slope keys")
    p.add_argument("--json", default="")
    a = p.parse_args()
    results, detail = [], {}
    for spec, kind in [(s, "ipc") for s in a.ipc] + [(s, "deme") for s in a.deme]:
        label, rest = spec.split("=", 1)
        parts = rest.split(":")
        rows = []
        for key in a.slopes.split(","):
            alpha = ALPHA[key]
            if kind == "ipc":
                tag, mu_k, mu_s = parts[0], float(parts[1]), float(parts[2])
                got = load_ipc(tag, key, parts[1])
                if got is None:
                    continue
                t, s, v, nrm, nw, loading = got
            else:
                pat, mu_k, mu_s, loading = parts[0], float(parts[1]), float(parts[2]), parts[3]
                got = load_deme(pat, key)
                if got is None:
                    continue
                t, s, v, nrm, nw, _ = got
            s_ref, v_ref = analytic(alpha, mu_k, mu_s, loading, t)
            rows.append((key, t, s, v, s_ref, v_ref, nrm, nw))
            detail.setdefault(label, {})[key] = dict(s_mm=float(s[-1] * 1e3), ref_mm=float(s_ref[-1] * 1e3),
                                                     v_mms=float(v[-1] * 1e3), normal_um=nrm)
        results.append(score(f"{kind}:{label} ({len(rows)} slopes)", rows))
    print(f"{'run':44s} {'wrong stick/slide':22s} {'|err| mean':>10s} {'max':>7s} {'creep mm':>9s} {'normal um':>10s} {'newton':>7s}")
    for r in results:
        print(f"{r['label']:44s} {','.join(r['wrong']) or '-':22s} {r['mean_abs_err'] * 100:9.1f}% {r['max_abs_err'] * 100:6.1f}% "
              f"{r['creep_mm']:9.2f} {r['normal_um']:10.0f} {r['newton']:7.2f}")
    if a.per_slope:
        keys = a.slopes.split(",")
        print("\nslid at 0.3 s (mm), analytic in brackets")
        print(f"{'run':16s} " + " ".join(f"{k:>14s}" for k in keys))
        for label, d in detail.items():
            print(f"{label:16s} " + " ".join(f"{d[k]['s_mm']:6.2f} ({d[k]['ref_mm']:5.2f})" if k in d else f"{'':>14s}" for k in keys))
    if a.json:
        json.dump(dict(summary=results, detail=detail), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
