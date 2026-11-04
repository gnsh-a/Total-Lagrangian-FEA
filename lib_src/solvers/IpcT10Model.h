#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

#include "lib_src/elements/FEAT10Data.cuh"

namespace tlfea {
struct ElasticEvaluation {
  double energy = 0, min_jacobian = 1;
  Eigen::VectorXd gradient;
  Eigen::SparseMatrix<double> hessian;
  bool valid = true;
};
// Evaluates the existing T10 material laws on CUDA with positive quadrature.
// Raw derivatives are kept separate from the projected Newton matrix.
class IpcT10Model {
 public:
  explicit IpcT10Model(GPU_FEAT10_Data& data);
  ~IpcT10Model();
  IpcT10Model(const IpcT10Model&)            = delete;
  IpcT10Model& operator=(const IpcT10Model&) = delete;
  const Eigen::SparseMatrix<double>& mass() const;
  const Eigen::MatrixXi& elements() const;
  const Eigen::MatrixXd& rest() const;
  Eigen::MatrixXd Positions() const;
  void SetPositions(const Eigen::MatrixXd& x);
  Eigen::VectorXd ExternalForce() const;
  ElasticEvaluation Evaluate(const Eigen::MatrixXd& x, bool derivatives = true,
                             bool project_psd = true);
  double FeasibleStep(const Eigen::MatrixXd& from, const Eigen::MatrixXd& to,
                      double min_det = 1e-8) const;
  void Constraints(Eigen::VectorXd& c, Eigen::SparseMatrix<double>& J);
  GPU_FEAT10_Data& data();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tlfea
