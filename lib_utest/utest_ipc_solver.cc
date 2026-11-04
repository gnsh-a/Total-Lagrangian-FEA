#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include "lib_src/constraints/FEAT10ConstraintManager.h"
#include "lib_src/solvers/IpcNewton.h"
#include "lib_utils/surface_trimesh_extract.h"
using namespace tlfea;
namespace {
Eigen::MatrixXd Tet(double z = 0) {
  Eigen::MatrixXd X(10, 3);
  X << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .5, 0, 0, .5, .5, 0, 0, .5, 0, 0, 0,
      .5, .5, 0, .5, 0, .5, .5;
  X.col(2).array() += z;
  return X;
}
struct Fixture {
  GPU_FEAT10_Data data;
  Eigen::MatrixXd X;
  Eigen::MatrixXi T;
  Fixture(int bodies = 1, bool mr = false)
      : data(bodies, 10 * bodies), X(10 * bodies, 3), T(bodies, 10) {
    for (int b = 0; b < bodies; ++b) {
      X.middleRows(10 * b, 10) = Tet(1.05 * b);
      for (int a = 0; a < 10; ++a)
        T(b, a) = 10 * b + a;
    }
    data.Initialize();
    data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
               Quadrature::tet5pt_weights, X.col(0), X.col(1), X.col(2), T);
    if (mr)
      data.ApplyMaterial(
          SolidMaterialProperties::MooneyRivlin(100, 50, 1000, 1, 0, 0));
    else
      data.ApplyMaterial(SolidMaterialProperties::SVK(1000, .3, 1, 0, 0));
    data.SetExternalForce(Eigen::VectorXd::Zero(30 * bodies));
  }
  ~Fixture() {
    data.Destroy();
  }
  std::unique_ptr<contact::IpcContact> Contact(double dhat = .1,
                                               double mu   = 0) {
    std::vector<ANCFCPUUtils::SurfaceTriMesh> meshes;
    for (int b = 0; b < T.rows(); ++b)
      meshes.push_back(
          ANCFCPUUtils::ExtractSurfaceTriMesh(X, T, {10 * b, b, 10, 1, "tet"}));
    contact::Config c;
    c.dhat      = dhat;
    c.stiffness = 10;
    c.mu_static = c.mu_kinetic = mu;
    return std::make_unique<contact::IpcContact>(
        contact::Surface::FromMeshes(X, meshes), c);
  }
};
Eigen::VectorXd Flat(const Eigen::MatrixXd& x) {
  return Eigen::Map<const Eigen::VectorXd>(
      Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>(x)
          .eval()
          .data(),
      x.size());
}
}  // namespace
TEST(IpcModel, MaterialEnergyGradientHessian) {
  for (bool mr : {false, true}) {
    Fixture f(1, mr);
    IpcT10Model model(f.data);
    auto X = f.X;
    X.col(0) *= 1.03;
    X(4, 2) += .005;
    const auto r = model.Evaluate(X, true, false);
    ASSERT_TRUE(r.valid);
    Eigen::VectorXd fd(30);
    Eigen::MatrixXd H(30, 30);
    const double eps = 1e-6;
    for (int j = 0; j < 30; ++j) {
      auto a = X, b = X;
      a(j / 3, j % 3) += eps;
      b(j / 3, j % 3) -= eps;
      fd[j] =
          (model.Evaluate(a, false).energy - model.Evaluate(b, false).energy) /
          (2 * eps);
      H.col(j) = (model.Evaluate(a, true, false).gradient -
                  model.Evaluate(b, true, false).gradient) /
                 (2 * eps);
    }
    EXPECT_LT((fd - r.gradient).norm() / r.gradient.norm(), 1e-6);
    EXPECT_LT((H - Eigen::MatrixXd(r.hessian)).norm() / H.norm(), 1e-6);
  }
}
TEST(IpcModel, MaterialPathGuardDetectsInteriorInversion) {
  Fixture f;
  IpcT10Model m(f.data);
  auto end = f.X;
  end.col(0) *= -2;
  end.col(1) *= -.5;
  EXPECT_TRUE(m.Evaluate(end, false).valid);
  const double a = m.FeasibleStep(f.X, end);
  EXPECT_GT(a, 0);
  EXPECT_LT(a, 1. / 3);
}
TEST(IpcSolver, RigidTranslationBEAndBDF2) {
  for (auto integrator : {IpcIntegrator::BackwardEuler, IpcIntegrator::BDF2}) {
    Fixture f;
    IpcNewtonParams params;
    params.integrator = integrator;
    IpcNewtonSolver solver(f.data, nullptr, params);
    Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(10, 3);
    velocity.col(0).setConstant(.2);
    solver.SetInitialVelocity(Flat(velocity));
    for (double h : {.001, .001, .0015}) {
      auto r = solver.TryStep(h);
      ASSERT_TRUE(r.accepted) << r.reason;
    }
    IpcT10Model model(f.data);
    EXPECT_LT((model.Positions() - f.X - solver.time() * velocity).norm(),
              1e-8);
    EXPECT_LT((solver.Velocity() - Flat(velocity)).norm(), 1e-7);
  }
}
TEST(IpcSolver, FailureRestoresStateLoadsTargetsAndHistory) {
  Fixture f;
  IpcNewtonParams p;
  p.max_newton  = 1;
  p.max_retries = 0;
  IpcNewtonSolver solver(f.data, nullptr, p);
  solver.SetStepInputs([&](double) {
    IpcStepInputs in;
    in.external_force     = Eigen::VectorXd::Constant(30, 10);
    in.constraint_targets = f.X.array() + 1;
    return in;
  });
  auto result = solver.Advance(.001);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(solver.time(), 0);
  EXPECT_EQ(solver.Velocity().norm(), 0);
  IpcT10Model model(f.data);
  EXPECT_EQ((model.Positions() - f.X).norm(), 0);
  EXPECT_EQ(model.ExternalForce().norm(), 0);
  Eigen::VectorXd x(10);
  cudaMemcpy(x.data(), f.data.GetX12JacDevicePtr(), 10 * sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_EQ((x - f.X.col(0)).norm(), 0);
}
TEST(IpcSolver, TwoFreeBodiesConserveMomentum) {
  Fixture f(2);
  IpcT10Model model(f.data);
  IpcNewtonParams p;
  p.gradient_atol = 1e-8;
  p.gradient_rtol = 1e-8;
  IpcNewtonSolver solver(f.data, f.Contact(), p);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(60);
  for (int i = 10; i < 20; ++i)
    v[3 * i + 2] = -.2;
  solver.SetInitialVelocity(v);
  auto momentum = [&](const Eigen::VectorXd& velocity) {
    const Eigen::VectorXd mv = model.mass() * velocity;
    Eigen::Vector3d r        = Eigen::Vector3d::Zero();
    for (int i = 0; i < 20; ++i)
      r += mv.segment<3>(3 * i);
    return r;
  };
  const auto initial = momentum(v);
  for (int i = 0; i < 3; ++i) {
    auto r = solver.TryStep(.001);
    ASSERT_TRUE(r.accepted) << r.reason;
    EXPECT_GT(r.contacts, 0u);
  }
  EXPECT_LT((momentum(solver.Velocity()) - initial).norm(), 1e-8);
}
TEST(IpcSolver, ZeroDampingIsEnforced) {
  Fixture f;
  f.data.ApplyMaterial(SolidMaterialProperties::SVK(1000, .3, 1, 1, 0));
  EXPECT_THROW(IpcT10Model model(f.data), std::invalid_argument);
}

TEST(IpcSolver, FixedAndPrescribedNodes) {
  Fixture f;
  Eigen::VectorXi fixed(3);
  fixed << 0, 1, 2;
  f.data.SetNodalFixed(fixed);
  IpcNewtonParams params;
  params.constraint_tolerance = 1e-8;
  IpcNewtonSolver solver(f.data, nullptr, params);
  solver.SetStepInputs([&](double t) {
    IpcStepInputs in;
    in.external_force     = Eigen::VectorXd::Zero(30);
    in.constraint_targets = f.X;
    in.constraint_targets.col(0).array() += .1 * t;
    return in;
  });
  auto result = solver.TryStep(.001);
  ASSERT_TRUE(result.accepted) << result.reason;
  IpcT10Model model(f.data);
  auto X = model.Positions();
  for (int n : {0, 1, 2})
    EXPECT_LT((X.row(n) - f.X.row(n) - Eigen::RowVector3d(.0001, 0, 0)).norm(),
              2e-8);
}

TEST(IpcSolver, AdaptiveRetryReevaluatesTimeDependentLoads) {
  Fixture f;
  IpcNewtonParams params;
  params.max_newton  = 12;
  params.max_retries = 4;
  IpcNewtonSolver solver(f.data, nullptr, params);
  int calls = 0;
  solver.SetStepInputs([&](double t) {
    ++calls;
    if (calls == 1)
      throw std::runtime_error("injected attempt failure");
    IpcStepInputs in;
    in.external_force = Eigen::VectorXd::Zero(30);
    EXPECT_LE(t, .001);
    return in;
  });
  const auto r = solver.Advance(.001);
  ASSERT_TRUE(r.accepted) << r.reason;
  EXPECT_EQ(r.retries, 1);
  EXPECT_DOUBLE_EQ(solver.time(), .001);
  EXPECT_GE(calls, 3);
}

TEST(IpcModel, PerBodyDensityAndElasticParameters) {
  Fixture f(2);
  f.data.ApplyMaterialsByElementRanges(
      {0, 1}, {1, 1},
      {SolidMaterialProperties::SVK(1000, .3, 1, 0, 0),
       SolidMaterialProperties::SVK(2000, .3, 3, 0, 0)});
  IpcT10Model model(f.data);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(60), v = u;
  u.head(30).setOnes();
  v.tail(30).setOnes();
  EXPECT_NEAR(v.dot(model.mass() * v) / u.dot(model.mass() * u), 3, 1e-12);
  auto X = f.X;
  X.col(0) *= 1.01;
  auto r = model.Evaluate(X, true, false);
  EXPECT_NEAR(r.gradient.tail(30).norm() / r.gradient.head(30).norm(), 2,
              1e-10);
}

TEST(IpcSolver, LaggedFrictionCouplesBothBodies) {
  Fixture f(2);
  IpcT10Model model(f.data);
  IpcNewtonParams p;
  p.max_outer = 30;
  IpcNewtonSolver solver(f.data, f.Contact(.1, .3), p);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(60);
  for (int i = 10; i < 20; ++i)
    v[3 * i] = .1;
  solver.SetInitialVelocity(v);
  auto total = [&](const Eigen::VectorXd& velocity) {
    Eigen::VectorXd mv = model.mass() * velocity;
    Eigen::Vector3d r  = Eigen::Vector3d::Zero();
    for (int i = 0; i < 20; ++i)
      r += mv.segment<3>(3 * i);
    return r;
  };
  const auto before = total(v);
  auto r            = solver.TryStep(.001);
  ASSERT_TRUE(r.accepted) << r.reason;
  EXPECT_LT((total(solver.Velocity()) - before).norm(), 1e-8);
  EXPECT_GT(solver.Velocity().head(30).norm(), 0);
}

TEST(IpcSolver, NonlinearJointUsesExistingConstraintManager) {
  Fixture f;
  FEAT10ConstraintManager manager(&f.data);
  auto point = [](int n) {
    FEAT10ConstraintPoint p;
    p.element_idx = 0;
    p.shape.setZero();
    p.shape[n] = 1;
    return p;
  };
  manager.AddDP1Constraint(point(0), point(1), point(0), point(2), 0);
  manager.AddNodeToWorldCD(0);
  manager.Finalize();
  Eigen::VectorXd load = Eigen::VectorXd::Zero(30);
  load[4]              = .01;
  f.data.SetExternalForce(load);
  IpcNewtonParams p;
  p.constraint_tolerance = 1e-9;
  IpcNewtonSolver solver(f.data, nullptr, p);
  auto result = solver.TryStep(.001);
  ASSERT_TRUE(result.accepted) << result.reason;
  IpcT10Model model(f.data);
  Eigen::VectorXd c;
  Eigen::SparseMatrix<double> J;
  model.Constraints(c, J);
  EXPECT_LT(c.lpNorm<Eigen::Infinity>(), 1e-9);
}

TEST(IpcSolver, GeneralPrescribedTargetsAndRollback) {
  Fixture f;
  FEAT10ConstraintManager manager(&f.data);
  auto point = [](int n) {
    FEAT10ConstraintPoint p;
    p.element_idx = 0;
    p.shape.setZero();
    p.shape[n] = 1;
    return p;
  };
  manager.AddDP1Constraint(point(0), point(1), point(0), point(2), 0);
  manager.AddNodeToWorldCD(0);
  manager.Finalize();
  const auto initial = f.data.GeneralConstraintTargets();
  ASSERT_GT(initial.size(), 0);
  IpcNewtonSolver solver(f.data, nullptr);
  solver.SetStepInputs([&](double t) {
    IpcStepInputs in;
    in.external_force             = Eigen::VectorXd::Zero(30);
    in.general_constraint_targets = initial;
    in.general_constraint_targets[0] += .01 * t;
    return in;
  });
  auto result = solver.TryStep(.001);
  ASSERT_TRUE(result.accepted) << result.reason;
  EXPECT_NEAR(f.data.GeneralConstraintTargets()[0], initial[0] + 1e-5, 1e-15);
  const auto saved    = f.data.GeneralConstraintTargets();
  const auto velocity = solver.Velocity();
  IpcNewtonParams params;
  params.max_newton = 1;
  solver.SetParameters(&params);
  solver.SetStepInputs([&](double) {
    IpcStepInputs in;
    in.external_force             = Eigen::VectorXd::Constant(30, 10);
    in.general_constraint_targets = saved.array() + 1;
    return in;
  });
  result = solver.TryStep(.001);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ((f.data.GeneralConstraintTargets() - saved).norm(), 0);
  EXPECT_EQ((solver.Velocity() - velocity).norm(), 0);
  EXPECT_DOUBLE_EQ(solver.time(), .001);
}

TEST(IpcSolver, NewContactsRebuildSymbolicPattern) {
  Fixture f(2);
  IpcNewtonParams params;
  params.max_newton = 100;
  params.time_step  = .01;
  IpcNewtonSolver solver(f.data, f.Contact(.02), params);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(60);
  for (int i = 10; i < 20; ++i)
    velocity[3 * i + 2] = -.5;
  solver.SetInitialVelocity(velocity);
  int analyses      = 0;
  bool contact_seen = false;
  for (int i = 0; i < 9; ++i) {
    auto result = solver.TryStep(.01);
    ASSERT_TRUE(result.accepted) << result.reason;
    analyses += result.symbolic_analyses;
    contact_seen = contact_seen || result.contacts > 0;
  }
  EXPECT_TRUE(contact_seen);
  EXPECT_GT(analyses, 1);
}

TEST(IpcSurface, RejectsInvalidAndNonconformingTopology) {
  auto X = Tet();
  Eigen::MatrixXi T(1, 10);
  for (int i = 0; i < 10; ++i)
    T(0, i) = i;
  const ANCFCPUUtils::MeshInstance single{0, 0, 10, 1, "tet"};
  const auto s = ANCFCPUUtils::ExtractSurfaceTriMesh(X, T, single);
  EXPECT_EQ(s.vertices.size(), 10u);
  EXPECT_EQ(s.triangles.size(), 16u);
  auto invalid  = T;
  invalid(0, 4) = 10;
  EXPECT_THROW(ANCFCPUUtils::ExtractSurfaceTriMesh(X, invalid, single),
               std::invalid_argument);
  Eigen::MatrixXi duplicate(2, 10);
  duplicate.row(0) = duplicate.row(1) = T.row(0);
  EXPECT_THROW(ANCFCPUUtils::ExtractSurfaceTriMesh(X, duplicate,
                                                   {0, 0, 10, 2, "duplicate"}),
               std::invalid_argument);
  X.conservativeResize(14, 3);
  X.row(10) << 0, 0, -1;
  X.row(11) = .5 * (X.row(0) + X.row(10));
  X.row(12) = .5 * (X.row(1) + X.row(10));
  X.row(13) = .5 * (X.row(2) + X.row(10));
  duplicate.row(1) << 0, 2, 1, 10, 6, 5, 4, 11, 13, 12;
  EXPECT_EQ(
      ANCFCPUUtils::ExtractSurfaceTriMesh(X, duplicate, {0, 0, 14, 2, "pair"})
          .triangles.size(),
      24u);
  duplicate(1, 4) = 3;
  EXPECT_THROW(ANCFCPUUtils::ExtractSurfaceTriMesh(
                   X, duplicate, {0, 0, 14, 2, "nonconforming"}),
               std::invalid_argument);
}
