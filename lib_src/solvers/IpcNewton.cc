#include "IpcNewton.h"

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace tlfea {
namespace {
using Sparse    = Eigen::SparseMatrix<double>;
using RowSparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
Eigen::VectorXd Flatten(const Eigen::MatrixXd& X) {
  return Eigen::Map<const Eigen::VectorXd>(
      Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>(X)
          .eval()
          .data(),
      X.size());
}
Eigen::MatrixXd Unflatten(const Eigen::VectorXd& x) {
  return Eigen::Map<
      const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
      x.data(), x.size() / 3, 3);
}
void Cuda(cudaError_t code) {
  if (code != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(code));
}
void Dss(cudssStatus_t code) {
  if (code != CUDSS_STATUS_SUCCESS)
    throw std::runtime_error("cuDSS failure " + std::to_string(int(code)));
}
void Validate(const IpcNewtonParams& p) {
  for (double v : {p.time_step, p.min_time_step, p.gradient_atol,
                   p.gradient_rtol, p.constraint_tolerance,
                   p.friction_tolerance, p.penalty, p.max_penalty, p.min_alpha})
    if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument(
          "IPC solver tolerances, steps and penalties must be finite and "
          "positive");
  if (p.min_time_step > p.time_step || p.max_penalty < p.penalty ||
      p.min_alpha >= 1 || p.max_newton < 1 || p.max_outer < 1 ||
      p.max_backtracks < 1 || p.max_retries < 0)
    throw std::invalid_argument("Invalid IPC solver limits");
}
// Own all descriptors and buffers together; changing the pattern invalidates
// symbolic and numerical factors before any buffer can be reallocated.
class DynamicCuDss {
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t cfg_    = nullptr;
  cudssData_t data_     = nullptr;
  cudssMatrix_t A_ = nullptr, B_ = nullptr, X_ = nullptr;
  int* rows_    = nullptr;
  int* cols_    = nullptr;
  double* vals_ = nullptr;
  double* b_    = nullptr;
  double* x_    = nullptr;
  std::vector<int> host_rows_, host_cols_;
  void ClearMatrix() {
    if (A_)
      cudssMatrixDestroy(A_);
    if (B_)
      cudssMatrixDestroy(B_);
    if (X_)
      cudssMatrixDestroy(X_);
    A_ = B_ = X_ = nullptr;
    if (data_)
      cudssDataDestroy(handle_, data_);
    data_ = nullptr;
    cudaFree(rows_);
    cudaFree(cols_);
    cudaFree(vals_);
    cudaFree(b_);
    cudaFree(x_);
    rows_ = cols_ = nullptr;
    vals_ = b_ = x_ = nullptr;
    host_rows_.clear();
    host_cols_.clear();
  }

 public:
  int analyses    = 0;
  size_t nonzeros = 0;
  DynamicCuDss() {
    Dss(cudssCreate(&handle_));
    Dss(cudssConfigCreate(&cfg_));
  }
  ~DynamicCuDss() {
    ClearMatrix();
    if (cfg_)
      cudssConfigDestroy(cfg_);
    if (handle_)
      cudssDestroy(handle_);
  }
  Eigen::VectorXd Solve(const Sparse& H, const Eigen::VectorXd& rhs) {
    RowSparse M = H.triangularView<Eigen::Upper>();
    M.makeCompressed();
    if (M.nonZeros() > INT_MAX || M.rows() > INT_MAX)
      throw std::overflow_error("IPC matrix exceeds cuDSS 32-bit indexing");
    const int n = M.rows(), nnz = M.nonZeros();
    nonzeros = nnz;
    std::vector<int> rows(M.outerIndexPtr(), M.outerIndexPtr() + n + 1),
        cols(M.innerIndexPtr(), M.innerIndexPtr() + nnz);
    const bool changed = rows != host_rows_ || cols != host_cols_;
    if (changed) {
      ClearMatrix();
      Cuda(cudaMalloc(&rows_, (n + 1) * sizeof(int)));
      Cuda(cudaMalloc(&cols_, nnz * sizeof(int)));
      Cuda(cudaMalloc(&vals_, nnz * sizeof(double)));
      Cuda(cudaMalloc(&b_, n * sizeof(double)));
      Cuda(cudaMalloc(&x_, n * sizeof(double)));
      Cuda(cudaMemcpy(rows_, rows.data(), (n + 1) * sizeof(int),
                      cudaMemcpyHostToDevice));
      Cuda(cudaMemcpy(cols_, cols.data(), nnz * sizeof(int),
                      cudaMemcpyHostToDevice));
      Dss(cudssDataCreate(handle_, &data_));
      Dss(cudssMatrixCreateCsr(&A_, n, n, nnz, rows_, nullptr, cols_, vals_,
                               CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                               CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER,
                               CUDSS_BASE_ZERO));
      Dss(cudssMatrixCreateDn(&B_, n, 1, n, b_, CUDSS_R_64F,
                              CUDSS_LAYOUT_COL_MAJOR));
      Dss(cudssMatrixCreateDn(&X_, n, 1, n, x_, CUDSS_R_64F,
                              CUDSS_LAYOUT_COL_MAJOR));
      host_rows_ = std::move(rows);
      host_cols_ = std::move(cols);
    }
    Cuda(cudaMemcpy(vals_, M.valuePtr(), nnz * sizeof(double),
                    cudaMemcpyHostToDevice));
    Cuda(
        cudaMemcpy(b_, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    if (changed) {
      Dss(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, cfg_, data_, A_, X_, B_));
      ++analyses;
    }
    Dss(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, cfg_, data_, A_, X_,
                     B_));
    Dss(cudssExecute(handle_, CUDSS_PHASE_SOLVE, cfg_, data_, A_, X_, B_));
    Eigen::VectorXd x(n);
    Cuda(cudaMemcpy(x.data(), x_, n * sizeof(double), cudaMemcpyDeviceToHost));
    if (!x.allFinite() ||
        (H * x - rhs).norm() > 1e-6 * std::max(1., rhs.norm()))
      throw std::runtime_error("IPC linear solve failed residual check");
    return x;
  }
};
struct PhaseTimes {
  double elastic = 0, contact = 0, ccd = 0, linear = 0, assembly = 0;
};
class Timer {
  double& total;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

 public:
  explicit Timer(double& value) : total(value) {}
  ~Timer() {
    total +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
  }
};
struct Objective {
  double energy = 0, min_det = 0;
  bool valid = true;
  Eigen::VectorXd g, c, contact_force;
  double active_min_distance = std::numeric_limits<double>::infinity();
  Sparse H;
  size_t contacts = 0;
};
}  // namespace
struct IpcNewtonSolver::Impl {
  IpcT10Model model;
  std::unique_ptr<contact::IpcContact> contact;
  IpcNewtonParams params;
  DynamicCuDss linear;
  PhaseTimes timings;
  Eigen::VectorXd v, v_old, lambda;
  Eigen::MatrixXd x_old;
  double now = 0, last_h = 0, rho;
  bool history = false;
  std::function<IpcStepInputs(double)> inputs;
  IpcStepResult last;
  Impl(GPU_FEAT10_Data& data, std::unique_ptr<contact::IpcContact> c,
       IpcNewtonParams p)
      : model(data), contact(std::move(c)), params(p), rho(p.penalty) {
    Validate(p);
    v      = Eigen::VectorXd::Zero(3 * data.get_n_coef());
    v_old  = v;
    x_old  = model.Positions();
    lambda = Eigen::VectorXd::Zero(data.get_n_constraint());
    if (contact)
      contact->ValidateInitial(x_old);
    if (!model.Evaluate(x_old, false).valid)
      throw std::invalid_argument(
          "IPC initial material configuration is invalid");
  }
  struct Snapshot {
    Eigen::MatrixXd x, x_old, targets;
    Eigen::VectorXd v, v_old, lambda, force, general_targets;
    double now, last_h, rho;
    bool history;
    contact::IpcContact::Checkpoint friction;
  };
  Snapshot Save() {
    auto& d     = model.data();
    const int n = d.get_n_coef();
    Eigen::MatrixXd t(n, 3);
    Cuda(cudaMemcpy(t.col(0).data(), d.GetX12JacDevicePtr(), n * sizeof(double),
                    cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(t.col(1).data(), d.GetY12JacDevicePtr(), n * sizeof(double),
                    cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(t.col(2).data(), d.GetZ12JacDevicePtr(), n * sizeof(double),
                    cudaMemcpyDeviceToHost));
    return {model.Positions(),
            x_old,
            t,
            v,
            v_old,
            lambda,
            model.ExternalForce(),
            d.GeneralConstraintTargets(),
            now,
            last_h,
            rho,
            history,
            contact ? contact->Save() : contact::IpcContact::Checkpoint{}};
  }
  void Restore(const Snapshot& s) {
    model.SetPositions(s.x);
    x_old   = s.x_old;
    v       = s.v;
    v_old   = s.v_old;
    lambda  = s.lambda;
    now     = s.now;
    last_h  = s.last_h;
    rho     = s.rho;
    history = s.history;
    model.data().UpdateConstraintTargets(s.targets.col(0), s.targets.col(1),
                                         s.targets.col(2));
    model.data().SetExternalForce(s.force);
    if (s.general_targets.size())
      model.data().UpdateGeneralConstraintTargets(s.general_targets);
    model.data().CalcP();
    Eigen::VectorXd c;
    Sparse J;
    model.Constraints(c, J);
    if (contact)
      contact->Restore(s.friction);
  }
  Objective Evaluate(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                     const Eigen::VectorXd& xhat, double a,
                     const Eigen::VectorXd& force, bool derivatives) {
    Objective o;
    const auto X = Unflatten(x);
    ElasticEvaluation elastic;
    {
      Timer timer(timings.elastic);
      elastic = model.Evaluate(X, derivatives);
    }
    o.min_det = elastic.min_jacobian;
    o.valid   = elastic.valid;
    if (!o.valid) {
      o.energy = INFINITY;
      return o;
    }
    const auto& M            = model.mass();
    const Eigen::VectorXd dx = x - xhat;
    // Drop the constant load potential at y to reduce cancellation in Armijo.
    o.energy =
        .5 * dx.dot(M * dx) / (a * a) + elastic.energy - force.dot(x - y);
    if (derivatives) {
      o.g = M * dx / (a * a) + elastic.gradient - force;
      o.H = M / (a * a) + elastic.hessian;
    }
    if (contact) {
      Timer timer(timings.contact);
      const auto normal = contact->Normal(X, derivatives);
      const auto friction =
          contact->Friction(Unflatten((x - y) / a), derivatives);
      o.contacts            = normal.collisions;
      o.active_min_distance = normal.active_min_distance;
      o.energy += normal.energy + a * friction.energy;
      if (derivatives) {
        o.g += normal.gradient + friction.gradient;
        o.contact_force = -(normal.gradient + friction.gradient);
        o.H += normal.hessian + friction.hessian / a;
      }
    }
    Timer assembly_timer(timings.assembly);
    Sparse J;
    model.Constraints(o.c, J);
    if (o.c.size() != lambda.size())
      throw std::runtime_error(
          "Constraint topology changed during an IPC solve");
    o.energy += lambda.dot(o.c) + .5 * rho * o.c.squaredNorm();
    if (derivatives && o.c.size()) {
      o.g += J.transpose() * (lambda + rho * o.c);
      o.H += rho * (J.transpose() * J);
    }
    o.valid = std::isfinite(o.energy) && (!derivatives || o.g.allFinite());
    return o;
  }
};
IpcNewtonSolver::IpcNewtonSolver(GPU_FEAT10_Data& d,
                                 std::unique_ptr<contact::IpcContact> c,
                                 IpcNewtonParams p)
    : impl_(std::make_unique<Impl>(d, std::move(c), p)) {}
IpcNewtonSolver::~IpcNewtonSolver() = default;
void IpcNewtonSolver::SetParameters(void* p) {
  if (!p)
    throw std::invalid_argument("Null IPC parameters");
  const auto value = *static_cast<IpcNewtonParams*>(p);
  Validate(value);
  impl_->params = value;
  impl_->rho    = value.penalty;
}
void IpcNewtonSolver::SetInitialVelocity(const Eigen::VectorXd& v) {
  if (v.size() != impl_->v.size() || !v.allFinite())
    throw std::invalid_argument("Invalid IPC velocity");
  impl_->v       = v;
  impl_->history = false;
}
void IpcNewtonSolver::SetStepInputs(std::function<IpcStepInputs(double)> f) {
  impl_->inputs = std::move(f);
}
Eigen::VectorXd IpcNewtonSolver::Velocity() const {
  return impl_->v;
}
double IpcNewtonSolver::time() const {
  return impl_->now;
}
const IpcStepResult& IpcNewtonSolver::last_result() const {
  return impl_->last;
}
IpcStepResult IpcNewtonSolver::TryStep(double h) {
  auto& p             = *impl_;
  const auto snapshot = p.Save();
  IpcStepResult result;
  result.time        = p.now;
  result.step_size   = h;
  const auto start   = std::chrono::steady_clock::now();
  const int analyses = p.linear.analyses;
  const auto timings = p.timings;
  try {
    if (!(h > 0) || !std::isfinite(h))
      throw std::invalid_argument("Invalid IPC time step");
    if (p.inputs) {
      auto in = p.inputs(p.now + h);
      if (in.external_force.size() != p.v.size() ||
          !in.external_force.allFinite())
        throw std::invalid_argument("Invalid step load");
      p.model.data().SetExternalForce(in.external_force);
      if (in.general_constraint_targets.size())
        p.model.data().UpdateGeneralConstraintTargets(
            in.general_constraint_targets);
      if (in.constraint_targets.size()) {
        if (p.model.data().GeneralConstraintTargets().size())
          throw std::invalid_argument(
              "Use scalar general_constraint_targets with general constraints");
        if (in.constraint_targets.rows() != snapshot.x.rows() ||
            in.constraint_targets.cols() != 3 ||
            !in.constraint_targets.allFinite())
          throw std::invalid_argument("Invalid step targets");
        p.model.data().UpdateConstraintTargets(in.constraint_targets.col(0),
                                               in.constraint_targets.col(1),
                                               in.constraint_targets.col(2));
      }
    }
    Eigen::VectorXd x = Flatten(snapshot.x), y = x, vhat = p.v;
    double a = h;
    if (p.params.integrator == IpcIntegrator::BDF2 && p.history) {
      const double r = h / p.last_h;
      // Large ratio jumps restart BE; variable-step BDF2 is zero-stable only
      // under a bounded growth ratio. Shrinking retries also restart below .5.
      if (r >= .5 && r <= 2) {
        const double a0 = (1 + 2 * r) / (1 + r), a1 = -(1 + r),
                     a2 = r * r / (1 + r);
        a               = h / a0;
        y               = (-a1 * x - a2 * Flatten(p.x_old)) / a0;
        vhat            = (-a1 * p.v - a2 * p.v_old) / a0;
      }
    }
    const Eigen::VectorXd xhat = y + a * vhat, force = p.model.ExternalForce();
    if (!force.allFinite())
      throw std::runtime_error("Nonfinite external force");
    if (p.contact) {
      p.contact->ValidateInitial(snapshot.x);
      p.contact->RefreshFriction(snapshot.x);
    }
    double previous_constraint = INFINITY;
    for (int outer = 0; outer < p.params.max_outer; ++outer) {
      result.outer_iterations = outer + 1;
      double initial_norm     = -1;
      bool converged          = false;
      for (int iter = 0; iter < p.params.max_newton; ++iter) {
        ++result.newton_iterations;
        auto o = p.Evaluate(x, y, xhat, a, force, true);
        if (!o.valid)
          throw std::runtime_error("Invalid IPC objective");
        const double norm = o.g.norm();
        if (p.params.verbose)
          std::cout << std::setprecision(17) << "  outer=" << outer
                    << " iter=" << iter << " energy=" << o.energy
                    << " gradient=" << norm << " contacts=" << o.contacts
                    << " gap=" << o.active_min_distance << std::endl;
        if (initial_norm < 0)
          initial_norm = norm;
        result.residual            = norm;
        result.energy              = o.energy;
        result.min_jacobian        = o.min_det;
        result.contacts            = o.contacts;
        result.active_min_distance = o.active_min_distance;
        if (norm <=
            p.params.gradient_atol + p.params.gradient_rtol * initial_norm) {
          converged = true;
          break;
        }
        Eigen::VectorXd direction;
        {
          Timer timer(p.timings.linear);
          direction = p.linear.Solve(o.H, -o.g);
        }
        double slope = o.g.dot(direction);
        if (!(slope < 0))
          throw std::runtime_error("IPC Newton direction is not descent");
        const auto from = Unflatten(x), to = Unflatten(x + direction);
        double alpha;
        {
          Timer timer(p.timings.ccd);
          alpha = p.model.FeasibleStep(from, to);
          if (p.contact)
            alpha = std::min(alpha, p.contact->StepSize(from, to));
        }
        bool accepted = false;
        for (int back = 0;
             back < p.params.max_backtracks && alpha >= p.params.min_alpha;
             ++back) {
          const Eigen::VectorXd trial = x + alpha * direction;
          auto t = p.Evaluate(trial, y, xhat, a, force, false);
          const double roundoff =
              32 * std::numeric_limits<double>::epsilon() *
              std::max({std::abs(o.energy), std::abs(t.energy), 1e-30});
          if (t.valid &&
              t.energy <= o.energy + 1e-4 * alpha * slope + roundoff) {
            if (p.params.verbose) {
              std::cout << "  accepted alpha=" << alpha
                        << " trial_energy=" << t.energy << std::endl;
            }
            x                = trial;
            accepted         = true;
            result.min_alpha = std::min(result.min_alpha, alpha);
            break;
          }
          alpha *= .5;
        }
        if (!accepted && p.params.verbose) {
          const Eigen::VectorXd unit = direction.normalized();
          const double eps           = 1e-7;
          const double fd =
              (p.Evaluate(x + eps * unit, y, xhat, a, force, false).energy -
               p.Evaluate(x - eps * unit, y, xhat, a, force, false).energy) /
              (2 * eps);
          std::cout << "  directional_fd=" << fd
                    << " analytic=" << o.g.dot(unit) << " alpha=" << alpha
                    << std::endl;
        }
        if (!accepted)
          throw std::runtime_error("IPC line search exhausted");
      }
      if (!converged)
        throw std::runtime_error("IPC Newton iteration limit");
      auto o                  = p.Evaluate(x, y, xhat, a, force, true);
      result.constraint_error = o.c.size() ? o.c.lpNorm<Eigen::Infinity>() : 0;
      double friction_change  = 0;
      if (p.contact && p.contact->config().mu_static > 0) {
        const auto velocity = Unflatten((x - y) / a);
        const auto before   = p.contact->Friction(velocity).gradient;
        p.contact->RefreshFriction(Unflatten(x));
        const auto after = p.contact->Friction(velocity).gradient;
        friction_change  = (after - before).norm() / std::max(1., after.norm());
      }
      if (p.params.verbose)
        std::cout << "  constraint=" << result.constraint_error
                  << " friction_change=" << friction_change << std::endl;
      if (result.constraint_error <= p.params.constraint_tolerance &&
          friction_change <= p.params.friction_tolerance) {
        // Physical path between committed states must also be feasible. This is
        // separate from the succession of Newton search directions.
        if (p.model.FeasibleStep(snapshot.x, Unflatten(x)) < 1 ||
            (p.contact && p.contact->StepSize(snapshot.x, Unflatten(x)) < 1))
          throw std::runtime_error(
              "Committed IPC trajectory is not collision/material feasible");
        p.model.SetPositions(Unflatten(x));
        p.model.data().CalcP();
        p.x_old   = snapshot.x;
        p.v_old   = p.v;
        p.v       = (x - y) / a;
        p.last_h  = h;
        p.history = true;
        p.now += h;
        result.contact_force       = o.contact_force;
        result.active_min_distance = o.active_min_distance;
        result.accepted            = true;
        result.time                = p.now;
        result.reason              = "converged";
        break;
      }
      if (o.c.size()) {
        p.lambda += p.rho * o.c;
        if (result.constraint_error > .5 * previous_constraint)
          p.rho = std::min(p.params.max_penalty, 10 * p.rho);
        previous_constraint = result.constraint_error;
      }
    }
    if (!result.accepted)
      throw std::runtime_error("IPC constraint/friction outer iteration limit");
  } catch (const std::exception& e) {
    p.Restore(snapshot);
    result.reason   = e.what();
    result.accepted = false;
  }
  result.elastic_seconds   = p.timings.elastic - timings.elastic;
  result.contact_seconds   = p.timings.contact - timings.contact;
  result.ccd_seconds       = p.timings.ccd - timings.ccd;
  result.linear_seconds    = p.timings.linear - timings.linear;
  result.assembly_seconds  = p.timings.assembly - timings.assembly;
  result.symbolic_analyses = p.linear.analyses - analyses;
  result.matrix_nonzeros   = p.linear.nonzeros;
  result.elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  p.last = result;
  return result;
}
IpcStepResult IpcNewtonSolver::Advance(double duration) {
  auto& p = *impl_;
  if (!(duration > 0) || !std::isfinite(duration))
    throw std::invalid_argument("Invalid IPC duration");
  const auto snapshot = p.Save();
  const double target = p.now + duration;
  if (!std::isfinite(target) || !(target > p.now))
    throw std::invalid_argument(
        "IPC interval cannot be represented at current time");
  double h    = std::min(duration, p.params.time_step);
  int retries = 0;
  IpcStepResult total;
  total.time      = p.now;
  total.step_size = duration;
  while (p.now < target) {
    h            = std::min(h, target - p.now);
    const auto r = TryStep(h);
    total.newton_iterations += r.newton_iterations;
    total.outer_iterations += r.outer_iterations;
    total.symbolic_analyses += r.symbolic_analyses;
    total.elapsed_seconds += r.elapsed_seconds;
    total.elastic_seconds += r.elastic_seconds;
    total.contact_seconds += r.contact_seconds;
    total.ccd_seconds += r.ccd_seconds;
    total.linear_seconds += r.linear_seconds;
    total.assembly_seconds += r.assembly_seconds;
    total.contacts            = r.contacts;
    total.active_min_distance = r.active_min_distance;
    total.contact_force       = r.contact_force;
    total.matrix_nonzeros     = r.matrix_nonzeros;
    total.residual            = r.residual;
    total.constraint_error    = r.constraint_error;
    total.min_jacobian        = r.min_jacobian;
    total.energy              = r.energy;
    total.min_alpha           = std::min(total.min_alpha, r.min_alpha);
    if (r.accepted) {
      if (target - p.now <= 1e-12 * duration) {
        p.now = target;
        break;
      }
      h = std::min(2 * h, p.params.time_step);
    } else {
      total.failed_attempts.push_back(
          r.reason + "; residual=" + std::to_string(r.residual));
      ++retries;
      h *= .5;
      if (retries > p.params.max_retries || h < p.params.min_time_step) {
        p.Restore(snapshot);
        total.reason  = r.reason;
        total.retries = retries;
        p.last        = total;
        return total;
      }
    }
  }
  total.accepted = true;
  total.time     = p.now;
  total.retries  = retries;
  total.reason   = "converged";
  p.last         = total;
  return total;
}
void IpcNewtonSolver::Solve() {
  auto r = Advance(impl_->params.time_step);
  if (!r.accepted)
    throw std::runtime_error("IPC step rolled back: " + r.reason);
}
}  // namespace tlfea
