/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    ExplicitStepGraph.cu
 * Brief:   CUDA graph runner for the explicit solvers.
 *==============================================================
 *==============================================================*/

#include "ExplicitStepGraph.cuh"

#include <chrono>

#include "../../lib_utils/cuda_utils.h"

ExplicitStepGraph::~ExplicitStepGraph() { Invalidate(); }

void ExplicitStepGraph::SetMode(ExplicitGraphMode mode) {
  if (mode != mode_) Invalidate();
  mode_ = mode;
}

void ExplicitStepGraph::Invalidate() {
  if (graphs_.empty()) return;
  HANDLE_ERROR(cudaDeviceSynchronize());  // no graph may still be running
  for (auto& kv : graphs_) HANDLE_ERROR(cudaGraphExecDestroy(kv.second));
  graphs_.clear();
}

void ExplicitStepGraph::Run(int n, cudaStream_t stream,
                            const std::function<void(cudaStream_t)>& step) {
  if (mode_ == ExplicitGraphMode::kOff) {
    for (int i = 0; i < n; ++i) step(stream);
    return;
  }
  while (n > 0) {
    const int k = n < kMaxUnrolled ? n : kMaxUnrolled;
    auto it     = graphs_.find(k);
    cudaGraphExec_t exec = it != graphs_.end() ? it->second : Build(k, stream, step);
    HANDLE_ERROR(cudaGraphLaunch(exec, stream));
    n -= k;
  }
}

cudaGraphExec_t ExplicitStepGraph::Build(
    int n, cudaStream_t stream, const std::function<void(cudaStream_t)>& step) {
  const auto t0 = std::chrono::steady_clock::now();
  cudaGraph_t graph;
  HANDLE_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < n; ++i) step(stream);
  HANDLE_ERROR(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec;
  HANDLE_ERROR(cudaGraphInstantiate(&exec, graph, 0));
  HANDLE_ERROR(cudaGraphDestroy(graph));
  graphs_[n] = exec;
  build_seconds_ += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
  return exec;
}
