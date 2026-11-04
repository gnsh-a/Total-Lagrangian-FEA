#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "lib_utils/surface_trimesh.h"

namespace tlfea::contact {
using Sparse = Eigen::SparseMatrix<double>;
struct Config {
  double dhat          = 1e-3;
  double dmin          = 0;
  double stiffness     = 1e5;
  double mu_static     = 0;
  double mu_kinetic    = 0;
  double eps_velocity  = 1e-3;
  double ccd_tolerance = 1e-8;
  bool self_collision  = false;
  bool area_weighting  = true;
};
// Scalar displacement map: surface x = rest + W*(full_x-full_rest).
// Solver DOFs and all derivative blocks use node-major xyz ordering.
struct Surface {
  Eigen::MatrixXd rest, full_rest;
  Eigen::MatrixXi faces;
  Sparse W;
  std::vector<int> body;
  std::vector<bool> kinematic;
  std::vector<std::pair<int, int>> excluded_body_pairs;
  static Surface FromMeshes(
      const Eigen::MatrixXd& full_rest,
      const std::vector<ANCFCPUUtils::SurfaceTriMesh>& meshes);
};
struct LocalBlock {
  std::vector<int> vertices;
  Eigen::VectorXd gradient;
  Eigen::MatrixXd hessian;
};
struct Evaluation {
  double energy = 0;
  // Minimum over active stencils; infinity means none in the activation band.
  double active_min_distance = std::numeric_limits<double>::infinity();
  Eigen::VectorXd gradient;
  Sparse hessian;
  std::vector<LocalBlock> blocks;
  size_t collisions = 0;
};
// CPU implementation behind a backend-neutral, mapped-local-block interface.
// No Toolkit types leak into CUDA translation units.
class IpcContact {
 public:
  struct Checkpoint {
    std::shared_ptr<const void> state;
  };
  IpcContact(Surface surface, Config config);
  ~IpcContact();
  IpcContact(const IpcContact&)            = delete;
  IpcContact& operator=(const IpcContact&) = delete;
  const Config& config() const;
  const Surface& surface() const;
  Eigen::MatrixXd Positions(const Eigen::MatrixXd& full_x) const;
  void ValidateInitial(const Eigen::MatrixXd& full_x) const;
  double StepSize(const Eigen::MatrixXd& from, const Eigen::MatrixXd& to) const;
  Evaluation Normal(const Eigen::MatrixXd& full_x, bool derivatives = true,
                    bool project_psd = true) const;
  Evaluation Friction(const Eigen::MatrixXd& full_velocity,
                      bool derivatives = true, bool project_psd = true) const;
  void RefreshFriction(const Eigen::MatrixXd& full_x);
  Checkpoint Save() const;
  void Restore(const Checkpoint& checkpoint);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tlfea::contact
