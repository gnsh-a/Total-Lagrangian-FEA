/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    ExplicitStepGraph.cu
 * Brief:   CUDA graph runner for the explicit solvers. The device
 *          loop sets its while-condition from a kernel, which needs
 *          the device runtime (rdc, cudadevrt).
 *==============================================================
 *==============================================================*/

#include "ExplicitStepGraph.cuh"

#include <chrono>

#include "../../lib_utils/cuda_utils.h"

// Device loop: the step count, written on the stream before each launch.
__global__ void explicitGraph_setCounter(int* counter, int n) { *counter = n; }

// Body tail of the device loop: one step done; stop when none remain.
__global__ void explicitGraph_decrement(cudaGraphConditionalHandle handle,
                                        int* counter) {
  if (--(*counter) <= 0) cudaGraphSetConditional(handle, 0);
}

ExplicitStepGraph::ExplicitStepGraph()
    : mode_(ExplicitGraphMode::kOff),
      device_graph_(nullptr),
      device_exec_(nullptr),
      d_counter_(nullptr),
      build_seconds_(0.0) {
  HANDLE_ERROR(cudaMalloc(&d_counter_, sizeof(int)));
}

ExplicitStepGraph::~ExplicitStepGraph() {
  Invalidate();
  if (d_counter_) cudaFree(d_counter_);
}

void ExplicitStepGraph::SetMode(ExplicitGraphMode mode) {
  if (mode != mode_) Invalidate();
  mode_ = mode;
}

void ExplicitStepGraph::Invalidate() {
  if (!unrolled_.empty() || device_exec_) {
    HANDLE_ERROR(cudaDeviceSynchronize());  // no graph may still be running
  }
  for (auto& kv : unrolled_) HANDLE_ERROR(cudaGraphExecDestroy(kv.second));
  unrolled_.clear();
  if (device_exec_) HANDLE_ERROR(cudaGraphExecDestroy(device_exec_));
  if (device_graph_) HANDLE_ERROR(cudaGraphDestroy(device_graph_));
  device_exec_  = nullptr;
  device_graph_ = nullptr;
}

void ExplicitStepGraph::Run(int n, cudaStream_t stream,
                            const std::function<void(cudaStream_t)>& step) {
  if (n <= 0) return;
  switch (mode_) {
    case ExplicitGraphMode::kOff:
      for (int i = 0; i < n; ++i) step(stream);
      break;
    case ExplicitGraphMode::kUnrolled:
      while (n > 0) {
        const int k = n < kMaxUnrolled ? n : kMaxUnrolled;
        auto it = unrolled_.find(k);
        cudaGraphExec_t exec =
            it != unrolled_.end() ? it->second : BuildUnrolled(k, stream, step);
        HANDLE_ERROR(cudaGraphLaunch(exec, stream));
        n -= k;
      }
      break;
    case ExplicitGraphMode::kDevice:
      if (!device_exec_) BuildDevice(stream, step);
      explicitGraph_setCounter<<<1, 1, 0, stream>>>(d_counter_, n);
      HANDLE_ERROR(cudaGraphLaunch(device_exec_, stream));
      break;
  }
}

cudaGraphExec_t ExplicitStepGraph::BuildUnrolled(
    int n, cudaStream_t stream, const std::function<void(cudaStream_t)>& step) {
  const auto t0 = std::chrono::steady_clock::now();
  cudaGraph_t graph;
  HANDLE_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < n; ++i) step(stream);
  HANDLE_ERROR(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec;
  HANDLE_ERROR(cudaGraphInstantiate(&exec, graph, 0));
  HANDLE_ERROR(cudaGraphDestroy(graph));
  unrolled_[n] = exec;
  build_seconds_ += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
  return exec;
}

void ExplicitStepGraph::BuildDevice(
    cudaStream_t stream, const std::function<void(cudaStream_t)>& step) {
  const auto t0 = std::chrono::steady_clock::now();
  HANDLE_ERROR(cudaGraphCreate(&device_graph_, 0));

  // While node; the condition is reset to 1 at every launch, and the counter
  // is written on the stream just before the launch.
  cudaGraphConditionalHandle handle;
  HANDLE_ERROR(cudaGraphConditionalHandleCreate(&handle, device_graph_, 1,
                                                cudaGraphCondAssignDefault));
  cudaGraphNodeParams cond = {};
  cond.type                = cudaGraphNodeTypeConditional;
  cond.conditional.handle  = handle;
  cond.conditional.type    = cudaGraphCondTypeWhile;
  cond.conditional.size    = 1;
  cudaGraphNode_t cond_node;
  HANDLE_ERROR(cudaGraphAddNode(&cond_node, device_graph_, nullptr, 0, &cond));
  cudaGraph_t body = cond.conditional.phGraph_out[0];

  // Body: one captured step, then the decrement.
  cudaGraph_t step_graph;
  HANDLE_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  step(stream);
  HANDLE_ERROR(cudaStreamEndCapture(stream, &step_graph));
  cudaGraphNode_t step_node;
  HANDLE_ERROR(
      cudaGraphAddChildGraphNode(&step_node, body, nullptr, 0, step_graph));
  HANDLE_ERROR(cudaGraphDestroy(step_graph));

  cudaKernelNodeParams dec = {};
  void* dec_args[]         = {&handle, &d_counter_};
  dec.func                 = reinterpret_cast<void*>(explicitGraph_decrement);
  dec.gridDim              = dim3(1);
  dec.blockDim             = dim3(1);
  dec.kernelParams         = dec_args;
  cudaGraphNode_t dec_node;
  HANDLE_ERROR(cudaGraphAddKernelNode(&dec_node, body, &step_node, 1, &dec));

  HANDLE_ERROR(cudaGraphInstantiate(&device_exec_, device_graph_, 0));
  build_seconds_ += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
}
