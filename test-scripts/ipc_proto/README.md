# IPC prototype: T10 sphere on a box

Python prototype of implicit IPC contact inside SyncedNewton's velocity-form Newton solve,
next to the explicit lagged coupling the DEME/hydro backends use today.

1. `sphere_on_box.py`: solver, scenes and CLI.
2. `suite.py`: every case used in the theory note (`--suite`).

## Run

Needs [uv](https://docs.astral.sh/uv/). Dependencies (numpy, scipy, matplotlib, ipctk 1.6.0) are
declared inline in the script.

```sh
uv run test-scripts/ipc_proto/sphere_on_box.py --scene rigid --contact ipc --integrator bdf2 --dt 1e-3
uv run test-scripts/ipc_proto/sphere_on_box.py --scene rigid --contact penalty --k-pen 1e9 --dt 5e-3
uv run test-scripts/ipc_proto/sphere_on_box.py --scene fem --dt 1e-3
uv run test-scripts/ipc_proto/sphere_on_box.py --suite --out results/
```

## What it solves

Per step, minimize over the end-of-step velocity `v` (`x = x_n + h v`):

```
Phi(v) = |v - v_n|_M^2 / (2h) + (Psi(x) - f_ext.x + kappa B(x)) / h + D(v)
R(v)   = M (v - v_n)/h + f_int(x) - f_ext + kappa grad B(x) + grad D(v)
H(v)   = M/h + h (K_t(x) + kappa [hess B(x)]+) + [hess D(v)]+
```

1. `R` and `H` without the contact terms are SyncedNewton's residual and Hessian (`solver_grad_L`).
2. Newton steps are capped by CCD, then backtracked on `Phi`. Every iterate is intersection-free.
3. `B` is the area-weighted clamped log barrier (Convergent IPC). `kappa` is in Pa and the gap
   at contact pressure `p` is about `2 kappa dhat / p`.
4. `D` is lagged smoothed Coulomb friction in velocity form.
5. BDF2 is the default integrator (`--integrator be` for backward Euler). It reuses the same
   solve with `x_n <- (4x_n - x_{n-1})/3`, `v_n <- (4v_n - v_{n-1})/3`, `h <- 2h/3`.
6. `--dhat` and `--kappa` are fixed per scene. The gap at contact pressure `p` is about
   `2 kappa dhat / p`: roughly 25 um at the impact peak and 0.66 mm at rest with the defaults.

## Scenes

1. `rigid`: the ptest sphere (`sphere_r0p15_low`) against an analytic box. The barrier, box
   distance and CCD are implemented here, per surface vertex.
2. `fem`: the same sphere against a T10 box with its bottom fixed, in one Newton system. Mesh-mesh
   contact comes from ipc-toolkit on the 4-way split of the T10 boundary faces.

`--contact penalty` (rigid scene only) applies a penalty pressure computed from start-of-step
positions and keeps it frozen in `f_ext` during the solve. This is the explicit coupling used
today.

## Limits

1. The rigid scene checks sphere vertices against the box only. It has no box-edge against
   sphere-edge pairs, which is fine for contact on a face.
2. It is CPU-only Python, so wall times say nothing about GPU cost.
3. Fixed nodes are eliminated rather than handled as ALM constraints. There is no Kelvin-Voigt
   damping.
4. The deformable-box minimum-distance diagnostic (`MeshIPC.min_distance`, 5 cm search radius) sometimes
   returns about 1e-11 m on the step right after separation, even single-threaded. The barrier's 1 mm
   query finds no pairs on those steps. Only the diagnostic is affected, and the cause is unknown.
