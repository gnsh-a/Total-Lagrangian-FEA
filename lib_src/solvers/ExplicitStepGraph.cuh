/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    ExplicitStepGraph.cuh
 * Brief:   Runs n explicit time steps on a stream, either as plain
 *          launches or as CUDA graphs of up to kMaxUnrolled steps.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <map>

enum class ExplicitGraphMode {
  kOff,      // plain kernel launches, no host sync between steps
  kUnrolled  // n steps captured into one graph (cached per n, up to kMaxUnrolled)
};

class ExplicitStepGraph {
 public:
  static constexpr int kMaxUnrolled = 1000;  // steps per graph

  ExplicitStepGraph() = default;
  ~ExplicitStepGraph();
  ExplicitStepGraph(const ExplicitStepGraph&) = delete;
  ExplicitStepGraph& operator=(const ExplicitStepGraph&) = delete;

  void SetMode(ExplicitGraphMode mode);
  ExplicitGraphMode Mode() const { return mode_; }

  // Drop all graphs; call when anything captured by value changes (dt, fixed
  // nodes, material constants passed as kernel arguments).
  void Invalidate();

  // Run n steps of `step` (which launches one step's kernels on its stream
  // argument) on `stream`. Returns without waiting for the GPU.
  void Run(int n, cudaStream_t stream,
           const std::function<void(cudaStream_t)>& step);

  // Wall time spent capturing and instantiating graphs so far (s).
  double BuildSeconds() const { return build_seconds_; }

 private:
  ExplicitGraphMode mode_ = ExplicitGraphMode::kUnrolled;
  std::map<int, cudaGraphExec_t> graphs_;  // steps -> graph
  double build_seconds_   = 0.0;

  cudaGraphExec_t Build(int n, cudaStream_t stream,
                        const std::function<void(cudaStream_t)>& step);
};
