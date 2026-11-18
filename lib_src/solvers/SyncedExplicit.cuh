#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    SyncedExplicit.cuh
 * Brief:   Declares the SyncedExplicitSolver class for GPU-accelerated
 *          symplectic Euler time integration with FEAT10 elements.
 *          Uses HRZ lumped mass for explicit dynamics.
 *==============================================================
 *==============================================================*/

#include <vector>

#include "../../lib_utils/cuda_utils.h"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "ExplicitStepGraph.cuh"
#include "SolverBase.h"

struct SyncedExplicitParams {
  double dt;  // Time step size
};

class SyncedExplicitSolver : public SolverBase {
 public:
  SyncedExplicitSolver(GPU_FEAT10_Data* element);
  ~SyncedExplicitSolver();

  // SolverBase interface
  void Solve() override;
  void SetParameters(void* params) override;

  // Advance by n time steps on the solver's stream, without waiting for the
  // GPU; CUDA calls on the default stream (copies to and from the host) order
  // after it.
  void SolveN(int n);

  // CUDA graphs of up to 1000 steps (default), or plain launches.
  void SetGraphMode(ExplicitGraphMode mode) { graph_.SetMode(mode); }
  double GraphBuildSeconds() const { return graph_.BuildSeconds(); }

  // Accessors for state
  double* d_velocity() { return d_vel_; }
  const double* d_velocity() const { return d_vel_; }
  double current_time() const { return current_time_; }
  int current_step() const { return current_step_; }
  int n_nodes() const { return n_nodes_; }
  
  // Boundary conditions
  void SetFixedNodes(const std::vector<int>& fixed_nodes);

  // Retrieve velocity to CPU
  void RetrieveVelocityToCPU(Eigen::VectorXd& vel_x, Eigen::VectorXd& vel_y,
                             Eigen::VectorXd& vel_z);

  // Reset simulation state
  void Reset();

 private:
  GPU_FEAT10_Data* element_;
  int n_nodes_;

  // Device memory
  double* d_vel_;           // [3 * n_nodes] velocity (vx, vy, vz interleaved per node)
  double* d_inv_mass_;      // [n_nodes] cached inverse lumped mass
  int* d_fixed_nodes_;      // [n_fixed] indices of fixed nodes
  int n_fixed_nodes_;

  // Parameters
  SyncedExplicitParams params_;
  double current_time_;
  int current_step_;

  // Internal state
  bool is_initialized_;
  bool inv_mass_ready_;

  // Stream and graphs; the graphs capture dt and the fixed-node array by value.
  cudaStream_t stream_;
  ExplicitStepGraph graph_;

  // Internal methods
  void AllocateMemory();
  void FreeMemory();
  void LaunchStep(cudaStream_t s);
};
