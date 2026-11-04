#pragma once
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "IpcT10Model.h"
#include "SolverBase.h"
#include "lib_src/contact/IpcContact.h"

namespace tlfea {
enum class IpcIntegrator { BackwardEuler, BDF2 };
struct IpcNewtonParams {
  double time_step = 1e-3, min_time_step = 1e-7;
  double gradient_atol = 1e-7, gradient_rtol = 1e-7,
         constraint_tolerance = 1e-7;
  double friction_tolerance = 1e-5, penalty = 1e6, max_penalty = 1e14;
  double min_alpha = 1e-12;
  int max_newton = 80, max_outer = 20, max_backtracks = 40, max_retries = 14;
  IpcIntegrator integrator = IpcIntegrator::BackwardEuler;
  bool verbose             = false;
};
struct IpcStepResult {
  bool accepted = false;
  std::string reason;
  double time = 0, step_size = 0, residual = 0, constraint_error = 0,
         min_jacobian = 0;
  double energy = 0, min_alpha = 1, elapsed_seconds = 0;
  double active_min_distance = std::numeric_limits<double>::infinity();
  Eigen::VectorXd contact_force;
  double elastic_seconds = 0, contact_seconds = 0, ccd_seconds = 0,
         linear_seconds = 0, assembly_seconds = 0;
  int newton_iterations = 0, outer_iterations = 0, retries = 0,
      symbolic_analyses = 0;
  size_t contacts = 0, matrix_nonzeros = 0;
  std::vector<std::string> failed_attempts;
};
struct IpcStepInputs {
  Eigen::VectorXd external_force;
  // Optional Nx3 Dirichlet targets; only fixed-node constraints consume these.
  Eigen::MatrixXd constraint_targets;
  // Optional scalar targets in FEAT10ConstraintManager's finalized row order.
  Eigen::VectorXd general_constraint_targets;
};
// One solve contains every contacting T10 body. Contact is evaluated on CPU;
// constitutive evaluation and sparse factorization retain CUDA/cuDSS.
class IpcNewtonSolver : public SolverBase {
 public:
  IpcNewtonSolver(GPU_FEAT10_Data& data,
                  std::unique_ptr<contact::IpcContact> contact,
                  IpcNewtonParams params = {});
  ~IpcNewtonSolver();
  IpcNewtonSolver(const IpcNewtonSolver&)            = delete;
  IpcNewtonSolver& operator=(const IpcNewtonSolver&) = delete;
  void SetParameters(void* params) override;
  void Solve()
      override;  // Throws if the requested output interval cannot finish.
  IpcStepResult TryStep(double h);
  IpcStepResult Advance(double duration);
  void SetInitialVelocity(const Eigen::VectorXd& velocity);
  void SetStepInputs(std::function<IpcStepInputs(double)> callback);
  Eigen::VectorXd Velocity() const;
  double time() const;
  const IpcStepResult& last_result() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tlfea
