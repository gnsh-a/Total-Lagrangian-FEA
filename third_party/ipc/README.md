# Pinned IPC Toolkit build

`repository.bzl` fetches only SHA-256 checked archives listed in `sources.json`.
`CMakeLists.txt` predefines the transitive dependency targets and disables network
fetches during compilation, Toolkit tests, Python, SIMD and GPU contact code.
The same Eigen headers/configuration are used by the surrounding repository.
Robust geometry/CCD retain strict floating-point behavior. TBB's shared runtime
is included in Bazel runfiles; no system TBB install is required.

The local edge-order patch fixes directional distance types being merged under
unordered edge keys. It changes the normal collision builder only. See
`docs/ipc/README.md` and the parallel-patch contact regression for rationale.
Toolkit remains optional at the target level; existing FE/DEME targets have no
new link dependency.

Upstream: https://github.com/ipc-sim/ipc-toolkit
Pinned source: https://github.com/ipc-sim/ipc-toolkit/tree/19bcab7e1f5a0d2addfcd28513422ff248ea504e
Third-party sources retain their upstream licenses inside the downloaded trees.
