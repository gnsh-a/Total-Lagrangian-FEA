#include <Eigen/Eigenvalues>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "IpcT10Model.h"
#include "lib_src/materials/MooneyRivlin.cuh"
#include "lib_src/materials/SVK.cuh"
#include "lib_utils/t10_quadrature.h"

namespace tlfea {
namespace {
void Cuda(cudaError_t code) {
  if (code != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(code));
}
struct QP {
  double G[10][3];
  double weight;
};
struct ElementResult {
  double energy, min_det, g[30], H[900];
  int valid;
};
__global__ void ReadMaterials(GPU_FEAT10_Data* d, FEAT10MeshMaterial* mats,
                              int* model) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < d->gpu_n_elem())
    mats[e] = d->mesh_material(e);
  if (e == 0)
    *model = d->material_model();
}
__global__ void ElasticKernel(GPU_FEAT10_Data* d, const QP* quadrature,
                              ElementResult* result, bool derivatives) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= d->gpu_n_elem())
    return;
  ElementResult& out = result[e];
  out.energy         = 0;
  out.min_det        = 1e100;
  out.valid          = 1;
  if (derivatives) {
    for (int i = 0; i < 30; ++i)
      out.g[i] = 0;
    for (int i = 0; i < 900; ++i)
      out.H[i] = 0;
  }
  double X[10][3];
  for (int a = 0; a < 10; ++a) {
    int n   = d->element_connectivity()(e, a);
    X[a][0] = d->x12()(n);
    X[a][1] = d->y12()(n);
    X[a][2] = d->z12()(n);
  }
  const auto mat = d->mesh_material(e);
  const bool mr  = d->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN;
  for (int q = 0; q < t10::kMassPoints; ++q) {
    const QP& p    = quadrature[e * t10::kMassPoints + q];
    double F[3][3] = {}, C[3][3] = {}, FFT[3][3] = {}, P[3][3], Fh[10][3] = {};
    for (int a = 0; a < 10; ++a)
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          F[i][j] += X[a][i] * p.G[a][j];
    const double J = mr_det3x3(F);
    out.min_det    = fmin(out.min_det, J);
    if (!(J > 1e-8) || !isfinite(J)) {
      out.valid  = 0;
      out.energy = INFINITY;
      return;
    }
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
          C[i][j] += F[k][i] * F[k][j];
          FFT[i][j] += F[i][k] * F[j][k];
        }
    const double I1 = C[0][0] + C[1][1] + C[2][2], trE = .5 * (I1 - 3);
    double Cnorm = 0, Enorm = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        Cnorm += C[i][j] * C[i][j];
        double E = .5 * (C[i][j] - (i == j));
        Enorm += E * E;
      }
    double A[3][3][3][3];
    if (mr) {
      const double jm23 = pow(J, -2. / 3), I2 = .5 * (I1 * I1 - Cnorm);
      out.energy += p.weight * (mat.mu10 * (jm23 * I1 - 3) +
                                mat.mu01 * (jm23 * jm23 * I2 - 3) +
                                .5 * mat.kappa * (J - 1) * (J - 1));
      if (derivatives) {
        mr_compute_P(F, mat.mu10, mat.mu01, mat.kappa, P);
        mr_compute_tangent_tensor(F, mat.mu10, mat.mu01, mat.kappa, A);
      }
    } else {
      out.energy += p.weight * (mat.mu * Enorm + .5 * mat.lambda * trE * trE);
      if (derivatives)
        svk_compute_P(F, mat.lambda, mat.mu, P);
    }
    if (!derivatives)
      continue;
    for (int a = 0; a < 10; ++a)
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          out.g[3 * a + i] += p.weight * P[i][j] * p.G[a][j];
          Fh[a][i] += F[i][j] * p.G[a][j];
        }
    for (int a = 0; a < 10; ++a)
      for (int b = 0; b < 10; ++b) {
        double K[3][3] = {};
        if (mr) {
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                  K[i][j] += p.weight * A[i][k][j][l] * p.G[a][k] * p.G[b][l];
        } else {
          double dot = 0, fdot = 0;
          for (int k = 0; k < 3; ++k) {
            dot += p.G[a][k] * p.G[b][k];
            fdot += Fh[a][k] * Fh[b][k];
          }
          svk_compute_tangent_block(Fh[a], Fh[b], dot, trE, fdot, FFT,
                                    mat.lambda, mat.mu, p.weight, K);
        }
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            out.H[(3 * a + i) * 30 + 3 * b + j] += K[i][j];
      }
  }
}
}  // namespace
struct IpcT10Model::Impl {
  GPU_FEAT10_Data& data;
  Eigen::MatrixXd rest;
  Eigen::MatrixXi T;
  Eigen::SparseMatrix<double> M;
  std::vector<QP> qp;
  QP* d_qp                = nullptr;
  ElementResult* d_result = nullptr;
  explicit Impl(GPU_FEAT10_Data& d) : data(d) {}
  ~Impl() {
    cudaFree(d_qp);
    cudaFree(d_result);
  }
};
IpcT10Model::IpcT10Model(GPU_FEAT10_Data& data)
    : impl_(std::make_unique<Impl>(data)) {
  auto& p     = *impl_;
  const int n = data.get_n_coef(), ne = data.get_n_elem();
  if (n <= 0 || ne <= 0 || n > INT_MAX / 3)
    throw std::invalid_argument(
        "IPC requires a nonempty T10 mesh with 32-bit DOFs");
  data.RetrieveConnectivityToCPU(p.T);
  Eigen::VectorXd x, y, z;
  data.RetrieveReferencePositionToCPU(x, y, z);
  p.rest.resize(n, 3);
  p.rest << x, y, z;
  if (!p.rest.allFinite())
    throw std::invalid_argument("Nonfinite T10 reference positions");
  FEAT10MeshMaterial* d_mat = nullptr;
  int* d_model              = nullptr;
  Cuda(cudaMalloc(&d_mat, ne * sizeof(FEAT10MeshMaterial)));
  Cuda(cudaMalloc(&d_model, sizeof(int)));
  ReadMaterials<<<(ne + 127) / 128, 128>>>(data.d_data, d_mat, d_model);
  std::vector<FEAT10MeshMaterial> mats(ne);
  int model;
  Cuda(cudaMemcpy(mats.data(), d_mat, ne * sizeof(FEAT10MeshMaterial),
                  cudaMemcpyDeviceToHost));
  Cuda(cudaMemcpy(&model, d_model, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_mat);
  cudaFree(d_model);
  if (model != MATERIAL_MODEL_MOONEY_RIVLIN && model != MATERIAL_MODEL_SVK)
    throw std::invalid_argument("Unsupported IPC material model");
  for (const auto& m : mats) {
    if (m.eta_damp != 0 || m.lambda_damp != 0)
      throw std::invalid_argument(
          "IPC initially supports zero material damping only");
    if (!(m.rho0 > 0) || !std::isfinite(m.rho0))
      throw std::invalid_argument("Invalid IPC density");
    if (model == MATERIAL_MODEL_MOONEY_RIVLIN) {
      if (!(std::isfinite(m.mu10) && std::isfinite(m.mu01) &&
            std::isfinite(m.kappa) && m.mu10 >= 0 && m.mu01 >= 0 &&
            m.mu10 + m.mu01 > 0 && m.kappa > 0))
        throw std::invalid_argument("Invalid MR material");
    } else if (!(std::isfinite(m.mu) && std::isfinite(m.lambda) && m.mu > 0 &&
                 3 * m.lambda + 2 * m.mu > 0))
      throw std::invalid_argument("Invalid SVK material");
  }
  p.qp.resize(ne * t10::kMassPoints);
  for (int e = 0; e < ne; ++e) {
    Eigen::Matrix<double, 10, 3> X;
    for (int a = 0; a < 10; ++a) {
      if (p.T(e, a) < 0 || p.T(e, a) >= n)
        throw std::invalid_argument("Invalid T10 connectivity");
      X.row(a) = p.rest.row(p.T(e, a));
    }
    for (int q = 0; q < t10::kMassPoints; ++q) {
      double xi, eta, zeta, w, N[10], G[10][3];
      t10::MassPoint(q, xi, eta, zeta, w);
      t10::Shape(xi, eta, zeta, N, G);
      Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor>> grad(&G[0][0]);
      const Eigen::Matrix3d J = X.transpose() * grad;
      if (!(J.determinant() > 0))
        throw std::invalid_argument("Invalid reference T10 Jacobian");
      auto& Q = p.qp[e * t10::kMassPoints + q];
      Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor>> phys(
          &Q.G[0][0]);
      phys     = grad * J.inverse();
      Q.weight = w * J.determinant();
    }
  }
  Cuda(cudaMalloc(&p.d_qp, p.qp.size() * sizeof(QP)));
  Cuda(cudaMemcpy(p.d_qp, p.qp.data(), p.qp.size() * sizeof(QP),
                  cudaMemcpyHostToDevice));
  Cuda(cudaMalloc(&p.d_result, ne * sizeof(ElementResult)));
  data.CalcDnDuPre();  // Populate legacy stress output at its own quadrature.
  data.CalcMassMatrix();
  std::vector<int> rows, cols;
  std::vector<double> vals;
  data.RetrieveMassCSRToCPU(rows, cols, vals);
  std::vector<Eigen::Triplet<double>> entries;
  for (int i = 0; i < n; ++i)
    for (int k = rows[i]; k < rows[i + 1]; ++k)
      for (int d = 0; d < 3; ++d)
        entries.emplace_back(3 * i + d, 3 * cols[k] + d, vals[k]);
  p.M.resize(3 * n, 3 * n);
  p.M.setFromTriplets(entries.begin(), entries.end());
}
IpcT10Model::~IpcT10Model() = default;
const Eigen::SparseMatrix<double>& IpcT10Model::mass() const {
  return impl_->M;
}
const Eigen::MatrixXi& IpcT10Model::elements() const {
  return impl_->T;
}
const Eigen::MatrixXd& IpcT10Model::rest() const {
  return impl_->rest;
}
GPU_FEAT10_Data& IpcT10Model::data() {
  return impl_->data;
}
Eigen::MatrixXd IpcT10Model::Positions() const {
  Eigen::VectorXd x, y, z;
  impl_->data.RetrievePositionToCPU(x, y, z);
  Eigen::MatrixXd X(x.size(), 3);
  X << x, y, z;
  return X;
}
void IpcT10Model::SetPositions(const Eigen::MatrixXd& x) {
  if (x.rows() != impl_->rest.rows() || x.cols() != 3 || !x.allFinite())
    throw std::invalid_argument("Invalid T10 positions");
  impl_->data.UpdatePositions(x.col(0), x.col(1), x.col(2));
}
Eigen::VectorXd IpcT10Model::ExternalForce() const {
  Eigen::VectorXd f;
  impl_->data.RetrieveExternalForceToCPU(f);
  return f;
}
ElasticEvaluation IpcT10Model::Evaluate(const Eigen::MatrixXd& x,
                                        bool derivatives, bool project) {
  SetPositions(x);
  auto& p      = *impl_;
  const int ne = p.T.rows();
  ElasticKernel<<<(ne + 31) / 32, 32>>>(p.data.d_data, p.d_qp, p.d_result,
                                        derivatives);
  Cuda(cudaGetLastError());
  std::vector<ElementResult> result(ne);
  Cuda(cudaMemcpy(result.data(), p.d_result, ne * sizeof(ElementResult),
                  cudaMemcpyDeviceToHost));
  ElasticEvaluation r;
  r.min_jacobian = std::numeric_limits<double>::infinity();
  if (derivatives) {
    r.gradient = Eigen::VectorXd::Zero(3 * x.rows());
    r.hessian.resize(3 * x.rows(), 3 * x.rows());
  }
  std::vector<Eigen::Triplet<double>> entries;
  for (int e = 0; e < ne; ++e) {
    const auto& a = result[e];
    r.energy += a.energy;
    r.min_jacobian = std::min(r.min_jacobian, a.min_det);
    r.valid        = r.valid && a.valid;
    if (!a.valid || !derivatives)
      continue;
    Eigen::Matrix<double, 30, 30> H =
        Eigen::Map<const Eigen::Matrix<double, 30, 30, Eigen::RowMajor>>(a.H);
    H = (.5 * (H + H.transpose())).eval();
    if (project) {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 30, 30>> eig(H);
      if (eig.info() != Eigen::Success)
        throw std::runtime_error("Elastic Hessian projection failed");
      H = eig.eigenvectors() * eig.eigenvalues().cwiseMax(0).asDiagonal() *
          eig.eigenvectors().transpose();
    }
    for (int i = 0; i < 30; ++i) {
      const int gi = 3 * p.T(e, i / 3) + i % 3;
      r.gradient[gi] += a.g[i];
      for (int j = 0; j < 30; ++j)
        entries.emplace_back(gi, 3 * p.T(e, j / 3) + j % 3, H(i, j));
    }
  }
  if (derivatives)
    r.hessian.setFromTriplets(entries.begin(), entries.end());
  r.valid = r.valid && std::isfinite(r.energy);
  return r;
}
// Along a straight nodal trajectory det(F) is cubic. Bound its first crossing
// using its derivative's stationary points and bisection on monotone intervals.
double IpcT10Model::FeasibleStep(const Eigen::MatrixXd& from,
                                 const Eigen::MatrixXd& to,
                                 double floor) const {
  double bound  = 1;
  const auto& p = *impl_;
  for (int e = 0; e < p.T.rows(); ++e)
    for (int q = 0; q < t10::kMassPoints; ++q) {
      const auto& Q     = p.qp[e * t10::kMassPoints + q];
      Eigen::Matrix3d A = Eigen::Matrix3d::Zero(), B = A;
      for (int a = 0; a < 10; ++a) {
        Eigen::Vector3d g(Q.G[a]);
        A += from.row(p.T(e, a)).transpose() * g.transpose();
        B += (to.row(p.T(e, a)) - from.row(p.T(e, a))).transpose() *
             g.transpose();
      }
      const double c0 = A.determinant() - floor, c3 = B.determinant();
      const double plus  = (A + B).determinant() - floor,
                   minus = (A - B).determinant() - floor;
      const double c2 = .5 * (plus + minus) - c0, c1 = .5 * (plus - minus) - c3;
      if (!(c0 > 0))
        return 0;
      auto value = [&](double t) { return ((c3 * t + c2) * t + c1) * t + c0; };
      std::vector<double> cuts = {0, bound};
      const double qa = 3 * c3, qb = 2 * c2, qc = c1;
      const double scale =
          std::max({std::abs(qa), std::abs(qb), std::abs(qc), 1e-300});
      if (std::abs(qa) < 1e-14 * scale) {
        if (std::abs(qb) > 1e-14 * scale) {
          double t = -qc / qb;
          if (t > 0 && t < bound)
            cuts.push_back(t);
        }
      } else {
        double disc = qb * qb - 4 * qa * qc;
        if (disc >= 0) {
          double u  = -.5 * (qb + std::copysign(std::sqrt(disc), qb));
          double t1 = u / qa, t2 = u == 0 ? -qb / (2 * qa) : qc / u;
          for (double t : {t1, t2})
            if (t > 0 && t < bound)
              cuts.push_back(t);
        }
      }
      std::sort(cuts.begin(), cuts.end());
      for (size_t i = 1; i < cuts.size(); ++i)
        if (value(cuts[i]) <= 0) {
          double lo = cuts[i - 1], hi = cuts[i];
          for (int k = 0; k < 60; ++k) {
            double mid = .5 * (lo + hi);
            if (value(mid) > 0)
              lo = mid;
            else
              hi = mid;
          }
          bound = std::min(bound, .9 * lo);
          break;
        }
    }
  return bound;
}
void IpcT10Model::Constraints(Eigen::VectorXd& c,
                              Eigen::SparseMatrix<double>& J) {
  auto& d     = impl_->data;
  const int m = d.get_n_constraint();
  c.resize(m);
  J.resize(m, 3 * d.get_n_coef());
  if (m == 0)
    return;
  d.CalcConstraintData();
  d.BuildConstraintJacobianCSR();
  Cuda(cudaMemcpy(c.data(), d.Get_Constraint_Ptr(), m * sizeof(double),
                  cudaMemcpyDeviceToHost));
  std::vector<int> rows, cols;
  std::vector<double> values;
  d.RetrieveConstraintJacobianCSRToCPU(rows, cols, values);
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < m; ++i)
    for (int k = rows[i]; k < rows[i + 1]; ++k)
      t.emplace_back(i, cols[k], values[k]);
  J.setFromTriplets(t.begin(), t.end());
}
}  // namespace tlfea
