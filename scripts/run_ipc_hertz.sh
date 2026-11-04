#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.8}"
OUTPUT_DIR="${IPC_LOG_DIR:-output/logs/ipc_hertz}"
mkdir -p "$OUTPUT_DIR"
bazel build -c opt //lib_bin/ipc:hertz_validation --jobs="${JOBS:-6}" 2>&1 | tee "$OUTPUT_DIR/build.log"
uv run scripts/ipc_hertz.py --out="$OUTPUT_DIR" "$@" 2>&1 | tee "$OUTPUT_DIR/run.log"
