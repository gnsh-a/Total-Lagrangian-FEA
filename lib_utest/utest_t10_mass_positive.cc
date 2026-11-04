#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cmath>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_utils/t10_quadrature.h"

TEST(T10Mass, ReferenceMonomialsThroughDegreeSeven) {
  for (int a = 0; a <= 7; ++a)
    for (int b = 0; b <= 7 - a; ++b)
      for (int c = 0; c <= 7 - a - b; ++c) {
        double value = 0;
        for (int q = 0; q < t10::kMassPoints; ++q) {
          double x, y, z, w;
          t10::MassPoint(q, x, y, z, w);
          ASSERT_GT(w, 0);
          value += w * std::pow(x, a) * std::pow(y, b) * std::pow(z, c);
        }
        const double exact = std::tgamma(a + 1) * std::tgamma(b + 1) *
                             std::tgamma(c + 1) / std::tgamma(a + b + c + 4);
        EXPECT_NEAR(value, exact, 2e-15);
      }
}

static Eigen::MatrixXd ReferenceNodes() {
  Eigen::MatrixXd X(10, 3);
  X << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .5, 0, 0, .5, .5, 0, 0, .5, 0, 0, 0,
      .5, .5, 0, .5, 0, .5, .5;
  return X;
}
static Eigen::MatrixXd Mass(GPU_FEAT10_Data& data) {
  data.CalcMassMatrix();
  std::vector<int> rows, cols;
  std::vector<double> values;
  data.RetrieveMassCSRToCPU(rows, cols, values);
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(10, 10);
  for (int i = 0; i < 10; ++i)
    for (int k = rows[i]; k < rows[i + 1]; ++k)
      M(i, cols[k]) = values[k];
  return M;
}
TEST(T10Mass, PositiveExactAndIndependentOfMovingTargets) {
  GPU_FEAT10_Data data(1, 10);
  data.Initialize();
  const auto X = ReferenceNodes();
  Eigen::MatrixXi T(1, 10);
  T << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
             Quadrature::tet5pt_weights, X.col(0), X.col(1), X.col(2), T);
  data.ApplyMaterial(SolidMaterialProperties::SVK(1000, .3, 1, 0, 0));
  const auto M = Mass(data);
  EXPECT_LT((M - M.transpose()).norm(), 1e-15);
  EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(M)
                .eigenvalues()
                .minCoeff(),
            1e-4);
  EXPECT_NEAR(M.sum(), 1. / 6, 1e-14);
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(M(i, i), 1. / 420, 1e-14);
  for (int i = 4; i < 10; ++i)
    EXPECT_NEAR(M(i, i), 4. / 315, 1e-14);
  data.UpdateConstraintTargets(2 * X.col(0), 2 * X.col(1), 2 * X.col(2));
  EXPECT_LT((Mass(data) - M).norm(), 1e-15);
  data.Destroy();
}
TEST(T10Mass, CurvedPositiveAndInvalidReferenceRejected) {
  for (bool inverted : {false, true}) {
    GPU_FEAT10_Data data(1, 10);
    data.Initialize();
    auto X = ReferenceNodes();
    X(4, 1) += .05;
    if (inverted)
      X.col(0) *= -1;
    Eigen::MatrixXi T(1, 10);
    T << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
               Quadrature::tet5pt_weights, X.col(0), X.col(1), X.col(2), T);
    data.ApplyMaterial(SolidMaterialProperties::SVK(1000, .3, 1, 0, 0));
    if (inverted)
      EXPECT_THROW(Mass(data), std::runtime_error);
    else
      EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Mass(data))
                    .eigenvalues()
                    .minCoeff(),
                0);
    data.Destroy();
  }
}
