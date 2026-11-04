# IPC implementation ledger

Branch `gnsh/ipc-implementation`, base `2fffca8`. CPU IPC Toolkit 1.6.1;
existing CUDA finite elements and cuDSS retained. GPU contact migration excluded.
Original prototype sources preserved in `test-scripts/ipc_proto` (outputs excluded).

The user selected a working implementation with documented issues, rather than
further solver tuning. Material damping remains zero and is explicitly enforced.

## Delivered scope

- [x] A: pinned optional Toolkit dependency; independent positive T10 mass
  quadrature; SVK/MR incremental energy and derivative checks.
- [x] B: persistent collision surface and displacement map, filtering, CPU
  backend, mapped local contributions, dynamic matrix graph lifecycle.
- [x] C: globally coupled T10 bodies, CCD and material feasibility, energy
  line search, transactional stepping and adaptive retries.
- [x] D: lagged friction, ALM constraints, prescribed trajectories.
- [x] E: BE/variable-step BDF2, per-body parameters,
  validation suite, reproducible demos, diagnostics and documentation.
- [x] Targeted regression checks of existing T10 mass, mesh, stress and cuDSS.

Implemented does not mean numerically tuned for every contact configuration.
Aligned flat-face friction remains a documented convergence issue. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) for reproduction and all scope limits,
[VALIDATION.md](VALIDATION.md) for evidence, and [README.md](README.md) for usage
and the numerical design.

## Invariants

Positions in the existing FE object are SoA; solver vectors are node-major AoS.
T10 boundary faces use four linear triangles, including all midside nodes.
Collision guarantees apply to this proxy, not the curved T10 surface.
Material validity at quadrature points is a separate check, not a global proof.
All contacting T10 bodies belong to one coupled solve. Failed attempts do not
advance physical state. Exhausted retries roll back the entire output interval.
