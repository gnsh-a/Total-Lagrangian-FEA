"""Cases used in the IPC write-up. Run through sphere_on_box.py --suite [--out DIR]."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace


def cases(Options):
    # Cases name their integrator explicitly (labels ending in -be use backward Euler).
    base = Options(quiet=True, integrator="be")
    out = []

    # A. Normal impact on a rigid box (no gravity, 1 m/s): step-size sweep.
    impact = replace(base, t_end=0.04)
    for dt in (1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3):
        out += [
            ("A", "ipc-be", replace(impact, dt=dt)),
            ("A", "ipc-bdf2", replace(impact, dt=dt, integrator="bdf2")),
            ("A", "pen1e9-be", replace(impact, dt=dt, contact="penalty", k_pen=1e9)),
            ("A", "pen1e8-be", replace(impact, dt=dt, contact="penalty", k_pen=1e8)),
        ]

    # B. Drop and come to rest under gravity (1 cm drop), large steps.
    for dt in (1e-3, 5e-3, 1e-2, 2e-2):
        rest = replace(base, dt=dt, gravity=9.81, v0=0.0, gap0=0.01, t_end=0.5)
        out += [
            ("B", "ipc-be", rest),
            ("B", "pen1e9-be", replace(rest, contact="penalty", k_pen=1e9)),
        ]

    # C. Deformable T10 box (bottom fixed), mesh-mesh Convergent IPC through ipctk.
    fem = replace(base, scene="fem", t_end=0.03)
    out += [
        ("C", "fem-ipc-be", replace(fem, dt=1e-3)),
        ("C", "fem-ipc-bdf2", replace(fem, dt=1e-3, integrator="bdf2")),
        ("C", "fem-ipc-bdf2", replace(fem, dt=2.5e-4, integrator="bdf2")),
        ("C", "fem-rest-be", replace(fem, dt=1e-2, gravity=9.81, v0=0.0, gap0=0.01, t_end=0.5)),
    ]

    # D. Oblique impact with friction (mu = 0.3): roll vs slide against rigid-body theory.
    for vx0 in (0.5, 1.0, 3.0):
        for lag in (1, 3):
            out.append(("D", f"fric-lag{lag}", replace(base, t_end=0.03, dt=5e-4, integrator="bdf2", mu=0.3, vx0=vx0, lag_iters=lag)))

    # E. Finer sphere mesh (contact patch resolved by more nodes).
    out.append(("E", "ipc-bdf2-med", replace(base, t_end=0.03, dt=1e-3, integrator="bdf2", mesh="data/meshes/ptest/sphere/sphere_r0p15_med.1")))
    return out


def _one(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    group, label, opt, run = args
    s, hist = run(opt)
    s.update(group=group, label=label, integrator=opt.integrator, vx0=opt.vx0, mu=opt.mu,
             lag_iters=opt.lag_iters, mesh=os.path.basename(opt.mesh), gravity=opt.gravity)
    return s, hist


def run_suite(run, Options, out_dir, workers=12):
    os.makedirs(out_dir, exist_ok=True)
    groups = os.environ.get("SUITE_GROUPS")
    jobs = [(g, l, o, run) for g, l, o in cases(Options) if not groups or g in groups.split(",")]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for f in as_completed(futs):
            g, l, o, _ = futs[f]
            try:
                s, hist = f.result()
            except Exception as e:  # keep the rest of the suite going
                s, hist = dict(group=g, label=l, dt=o.dt, error=repr(e)), []
            results.append(dict(summary=s, hist=hist))
            print(json.dumps(s), flush=True)
            json.dump(results, open(os.path.join(out_dir, "suite.json"), "w"))
    return results
