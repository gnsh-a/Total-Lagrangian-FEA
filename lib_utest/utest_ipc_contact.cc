#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include "lib_src/contact/IpcContact.h"
using namespace tlfea::contact;
static Surface Triangles() {
  Surface s;
  s.rest.resize(6, 3);
  s.rest << -1, -1, 0, 1, -1, 0, 0, 1, 0, .03, -.2, .03, .23, -.1, .04, .02, .2,
      .035;
  s.full_rest = s.rest;
  s.faces.resize(2, 3);
  s.faces << 0, 1, 2, 3, 4, 5;
  s.W.resize(6, 6);
  s.W.setIdentity();
  s.body = {0, 0, 0, 1, 1, 1};
  return s;
}
static Config Params() {
  Config c;
  c.dhat         = .1;
  c.stiffness    = 100;
  c.mu_static    = .3;
  c.mu_kinetic   = .3;
  c.eps_velocity = .01;
  return c;
}
TEST(IpcContact, BarrierDerivativesAndMomentumBalance) {
  auto s = Triangles();
  IpcContact c(s, Params());
  c.ValidateInitial(s.rest);
  const auto r = c.Normal(s.rest, true, false);
  ASSERT_GT(r.collisions, 0u);
  const double eps = 1e-6;
  Eigen::VectorXd fd(18);
  Eigen::MatrixXd H(18, 18);
  for (int j = 0; j < 18; ++j) {
    auto a = s.rest, b = s.rest;
    a(j / 3, j % 3) += eps;
    b(j / 3, j % 3) -= eps;
    fd[j] = (c.Normal(a, false).energy - c.Normal(b, false).energy) / (2 * eps);
    H.col(j) = (c.Normal(a, true, false).gradient -
                c.Normal(b, true, false).gradient) /
               (2 * eps);
  }
  EXPECT_LT((fd - r.gradient).norm() / r.gradient.norm(), 1e-7);
  EXPECT_LT((H - Eigen::MatrixXd(r.hessian)).norm() / H.norm(), 1e-6);
  Eigen::Vector3d total = Eigen::Vector3d::Zero();
  for (int i = 0; i < 6; ++i)
    total += r.gradient.segment<3>(3 * i);
  EXPECT_LT(total.norm(), 1e-9);
  const auto projected = c.Normal(s.rest);
  EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
                Eigen::MatrixXd(projected.hessian))
                .eigenvalues()
                .minCoeff(),
            -1e-7);
}
TEST(IpcContact, SweptCrossingThicknessAndFilters) {
  auto s      = Triangles();
  auto config = Params();
  config.dmin = .005;
  IpcContact c(s, config);
  auto end = s.rest;
  end.bottomRows(3).col(2).array() -= .1;
  const double a = c.StepSize(s.rest, end);
  EXPECT_GT(a, 0);
  EXPECT_LT(a, .35);
  c.ValidateInitial(s.rest + a * (end - s.rest));
  s.excluded_body_pairs = {{0, 1}};
  IpcContact excluded(s, config);
  EXPECT_DOUBLE_EQ(excluded.StepSize(s.rest, end), 1);
  EXPECT_EQ(excluded.Normal(s.rest).collisions, 0u);
}
TEST(IpcContact, FrictionDerivativesDissipationAndCheckpoint) {
  auto s = Triangles();
  IpcContact c(s, Params());
  c.RefreshFriction(s.rest);
  const auto saved  = c.Save();
  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(6, 3);
  v.bottomRows(3).col(0).setConstant(.05);
  v(4, 1) = .01;
  auto r  = c.Friction(v, true, false);
  ASSERT_GT(r.energy, 0);
  Eigen::VectorXd flat = Eigen::Map<const Eigen::VectorXd>(
      Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>(v)
          .eval()
          .data(),
      18);
  EXPECT_GT(flat.dot(r.gradient),
            0);  // Force = -gradient, hence nonpositive work.
  Eigen::VectorXd fd(18);
  Eigen::MatrixXd H(18, 18);
  const double eps = 1e-7;
  for (int j = 0; j < 18; ++j) {
    auto a = v, b = v;
    a(j / 3, j % 3) += eps;
    b(j / 3, j % 3) -= eps;
    fd[j] =
        (c.Friction(a, false).energy - c.Friction(b, false).energy) / (2 * eps);
    H.col(j) = (c.Friction(a, true, false).gradient -
                c.Friction(b, true, false).gradient) /
               (2 * eps);
  }
  EXPECT_LT((fd - r.gradient).norm() / r.gradient.norm(), 1e-7);
  EXPECT_LT((H - Eigen::MatrixXd(r.hessian)).norm() / H.norm(), 1e-6);
  auto far = s.rest;
  far.bottomRows(3).col(2).array() += 10;
  c.RefreshFriction(far);
  EXPECT_EQ(c.Friction(v).collisions, 0u);
  c.Restore(saved);
  EXPECT_DOUBLE_EQ(c.Friction(v).energy, r.energy);
}
TEST(IpcContact, WeightedMapChainRule) {
  auto s = Triangles();
  s.full_rest.conservativeResize(7, 3);
  s.full_rest.row(6) = s.rest.row(3);
  s.W.resize(6, 7);
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < 6; ++i)
    if (i != 3)
      t.emplace_back(i, i, 1);
  t.emplace_back(3, 3, .4);
  t.emplace_back(3, 6, .6);
  s.W.setFromTriplets(t.begin(), t.end());
  IpcContact c(s, Params());
  const auto r = c.Normal(s.full_rest, true, false);
  EXPECT_LT(
      (r.gradient.segment<3>(9) / .4 - r.gradient.segment<3>(18) / .6).norm(),
      1e-10);
  const double eps = 1e-6;
  auto a = s.full_rest, b = a;
  a(6, 2) += eps;
  b(6, 2) -= eps;
  EXPECT_NEAR(
      (c.Normal(a, false).energy - c.Normal(b, false).energy) / (2 * eps),
      r.gradient[20], 1e-5);
}
TEST(IpcContact, RejectInvalidGeometryAndParameters) {
  auto s   = Triangles();
  auto bad = Params();
  bad.dhat = -1;
  EXPECT_THROW(IpcContact(s, bad), std::invalid_argument);
  auto config = Params();
  config.dmin = .05;
  IpcContact c(s, config);
  EXPECT_THROW(c.ValidateInitial(s.rest), std::invalid_argument);
  s.faces(0, 1) = s.faces(0, 0);
  EXPECT_THROW(IpcContact(s, Params()), std::invalid_argument);
}

TEST(IpcContact, DistinctStaticKineticFrictionAtSmallSlip) {
  auto s            = Triangles();
  auto config       = Params();
  config.mu_static  = .5;
  config.mu_kinetic = .2;
  IpcContact c(s, config);
  c.RefreshFriction(s.rest);
  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(6, 3);
  v.bottomRows(3).col(0).setConstant(.004);
  const auto r = c.Friction(v, true, false);
  auto a = v, b = v;
  const double eps = 1e-7;
  a(3, 0) += eps;
  b(3, 0) -= eps;
  EXPECT_NEAR(
      (c.Friction(a, false).energy - c.Friction(b, false).energy) / (2 * eps),
      r.gradient[9], 1e-6);
  EXPECT_LT(((c.Friction(a, true, false).gradient -
              c.Friction(b, true, false).gradient) /
                 (2 * eps) -
             Eigen::MatrixXd(r.hessian).col(9))
                .norm(),
            1e-3);
}

TEST(IpcContact, SelfCollisionAndStaticObstacles) {
  auto s = Triangles();
  s.body.assign(6, 0);
  auto config = Params();
  IpcContact disabled(s, config);
  EXPECT_EQ(disabled.Normal(s.rest).collisions, 0u);
  config.self_collision = true;
  IpcContact enabled(s, config);
  EXPECT_GT(enabled.Normal(s.rest).collisions, 0u);
  s           = Triangles();
  s.kinematic = {true, true, true, false, false, false};
  s.W.setZero();
  std::vector<Eigen::Triplet<double>> w;
  for (int i = 3; i < 6; ++i)
    w.emplace_back(i, i, 1);
  s.W.setFromTriplets(w.begin(), w.end());
  IpcContact obstacle(s, Params());
  const auto r = obstacle.Normal(s.full_rest);
  EXPECT_EQ(r.gradient.head(9).norm(), 0);
  EXPECT_GT(r.gradient.tail(9).norm(), 0);
}

// Nearly parallel triangulated patches generate mollified edge-vertex stencils
// in both edge orders. Their directional distance types must be transposed when
// canonicalizing edge IDs (see the pinned Toolkit patch).
TEST(IpcContact, ParallelPatchesHaveRepeatablePotentialAndDerivatives) {
  Surface s;
  constexpr int n = 4, nv = n * n;
  s.rest.resize(2 * nv, 3);
  s.faces.resize(4 * (n - 1) * (n - 1), 3);
  int f = 0;
  for (int b = 0; b < 2; ++b)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        const int v = b * nv + i * n + j;
        s.rest.row(v) << .1 * i + b * .013, .1 * j + b * .021,
            b * (.001 + 1e-6 * i * j);
        s.body.push_back(b);
        if (i + 1 < n && j + 1 < n) {
          s.faces.row(f++) << v, v + 1, v + n;
          s.faces.row(f++) << v + 1, v + n + 1, v + n;
        }
      }
  s.full_rest = s.rest;
  s.W.resize(2 * nv, 2 * nv);
  s.W.setIdentity();
  auto config = Params();
  config.dhat = .003;
  IpcContact c(s, config);
  const auto reference = c.Normal(s.rest, true, false);
  ASSERT_GT(reference.collisions, 0u);
  for (int repeat = 0; repeat < 40; ++repeat) {
    const auto r = c.Normal(s.rest, true, false);
    EXPECT_EQ(r.collisions, reference.collisions);
    EXPECT_NEAR(r.energy, reference.energy, 1e-12 * reference.energy);
    EXPECT_LT((r.gradient - reference.gradient).norm(),
              1e-11 * reference.gradient.norm());
  }
  auto plus = s.rest, minus = s.rest;
  const double eps = 1e-8;
  plus(nv + n + 1, 2) += eps;
  minus(nv + n + 1, 2) -= eps;
  EXPECT_NEAR((c.Normal(plus, false).energy - c.Normal(minus, false).energy) /
                  (2 * eps),
              reference.gradient[3 * (nv + n + 1) + 2],
              1e-6 * reference.gradient.norm());
}

TEST(IpcContact, RejectsInitialSolidContainment) {
  Surface s;
  s.rest.resize(8, 3);
  s.rest << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .1, .1, .1, .2, .1, .1, .1, .2,
      .1, .1, .1, .2;
  s.full_rest = s.rest;
  s.W.resize(8, 8);
  s.W.setIdentity();
  s.body = {0, 0, 0, 0, 1, 1, 1, 1};
  s.faces.resize(8, 3);
  s.faces << 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3, 4, 6, 5, 4, 5, 7, 4, 7, 6, 5,
      6, 7;
  IpcContact contact(s, Params());
  EXPECT_THROW(contact.ValidateInitial(s.rest), std::invalid_argument);
  auto separate = s.rest;
  separate.bottomRows(4).array() += 2;
  EXPECT_NO_THROW(contact.ValidateInitial(separate));
}
