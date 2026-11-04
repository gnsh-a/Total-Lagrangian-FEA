#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.8}"
OUTPUT_DIR="${IPC_LOG_DIR:-output/logs/ipc_validation}"
mkdir -p "$OUTPUT_DIR"
bazel test -c opt //lib_utest:utest_t10_mass_positive //lib_utest:utest_ipc_contact //lib_utest:utest_ipc_solver \
  //lib_utest:utest_sparse_mass //lib_utest:utest_feat10_mesh //lib_utest:utest_feat10_von_mises //lib_utest:utest_feat10_cudss \
  --jobs="${JOBS:-6}" --test_output=errors --test_strategy=standalone 2>&1 | tee "$OUTPUT_DIR/tests.log"
bazel build -c opt //lib_bin/ipc:contact_demo --jobs="${JOBS:-6}" 2>&1 | tee "$OUTPUT_DIR/build.log"
for scenario in brick sphere two-body; do
  mkdir -p "$OUTPUT_DIR/$scenario"
  mu=.3
  if [[ "$scenario" == "two-body" ]]; then mu=0; fi
  ./bazel-bin/lib_bin/ipc/contact_demo --mu="$mu" --case="$scenario" --steps="${STEPS:-10}" --stiffness=1000 \
    --out="$OUTPUT_DIR/$scenario" 2>&1 | tee "$OUTPUT_DIR/$scenario/run.log"
done

mkdir -p "$OUTPUT_DIR/mr_bdf2"
./bazel-bin/lib_bin/ipc/contact_demo --case=two-body --material=mr --integrator=bdf2 --mu=0 \
  --steps="${STEPS:-10}" --stiffness=1000 --out="$OUTPUT_DIR/mr_bdf2" \
  2>&1 | tee "$OUTPUT_DIR/mr_bdf2/run.log"
