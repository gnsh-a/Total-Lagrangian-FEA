#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "lib_src/solvers/IpcNewton.h"
#include "lib_utils/cli_utils.h"
#include "lib_utils/mesh_manager.h"
#include "lib_utils/surface_trimesh_extract.h"

using namespace tlfea;
namespace {
struct Mesh {
  Eigen::MatrixXd X;
  Eigen::MatrixXi T;
};
Mesh Box(Eigen::Vector3d low, Eigen::Vector3d high, int resolution) {
  const int n = resolution + 1;
  std::vector<Eigen::Vector3d> vertices;
  auto id = [&](int i, int j, int k) { return (i * n + j) * n + k; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        vertices.push_back(
            low +
            (high - low).cwiseProduct(Eigen::Vector3d(i, j, k) / resolution));
  std::vector<std::array<int, 10>> tets;
  std::map<std::pair<int, int>, int> mids;
  const int perm[6][3]  = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                           {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  const int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  for (int i = 0; i < resolution; ++i)
    for (int j = 0; j < resolution; ++j)
      for (int k = 0; k < resolution; ++k)
        for (auto& order : perm) {
          Eigen::Vector3i p(i, j, k);
          std::array<int, 10> tet;
          tet[0] = id(p[0], p[1], p[2]);
          for (int a = 0; a < 3; ++a) {
            ++p[order[a]];
            tet[a + 1] = id(p[0], p[1], p[2]);
          }
          Eigen::Matrix3d J;
          for (int a = 0; a < 3; ++a)
            J.col(a) = vertices[tet[a + 1]] - vertices[tet[0]];
          if (J.determinant() < 0)
            std::swap(tet[1], tet[2]);
          for (int e = 0; e < 6; ++e) {
            int a = tet[edges[e][0]], b = tet[edges[e][1]];
            auto key = std::minmax(a, b);
            auto it  = mids.find(key);
            if (it == mids.end()) {
              int index = vertices.size();
              vertices.push_back(.5 * (vertices[a] + vertices[b]));
              mids[key]  = index;
              tet[4 + e] = index;
            } else
              tet[4 + e] = it->second;
          }
          tets.push_back(tet);
        }
  Mesh m;
  m.X.resize(vertices.size(), 3);
  m.T.resize(tets.size(), 10);
  for (size_t i = 0; i < vertices.size(); ++i)
    m.X.row(i) = vertices[i];
  for (size_t e = 0; e < tets.size(); ++e)
    for (int a = 0; a < 10; ++a)
      m.T(e, a) = tets[e][a];
  return m;
}
Eigen::VectorXd Flatten(const Eigen::MatrixXd& x) {
  return Eigen::Map<const Eigen::VectorXd>(
      Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>(x)
          .eval()
          .data(),
      x.size());
}
}  // namespace
int main(int argc, char** argv) {
  try {
    ANCFCPUUtils::Cli cli(argv[0]);
    cli.SetDescription(
        "CPU IPC contact with coupled CUDA T10 solids and cuDSS. Material "
        "damping is zero.");
    cli.AddString("case", "brick", "brick | sphere | two-body");
    cli.AddString("sphere_resolution", "low", "low | med | high");
    cli.AddInt("steps", 20, "output intervals");
    cli.AddInt("max_newton", 80, "Newton iterations per outer iteration");
    cli.AddInt("max_outer", 20, "constraint/friction outer iteration limit");
    cli.AddInt("max_retries", 14, "adaptive retry limit");
    cli.AddBool("verbose", false, "print nonlinear convergence diagnostics");
    cli.AddInt("resolution", 1, "box subdivisions per axis");
    cli.AddDouble("dt", .001, "output interval");
    cli.AddDouble("E", 1e5, "Young modulus");
    cli.AddDouble("density", 1000, "mass density");
    cli.AddDouble("dhat", .003, "barrier activation distance");
    cli.AddDouble("dmin", 0, "hard separation offset");
    cli.AddDouble("stiffness", 1e5, "physical barrier stiffness");
    cli.AddDouble("mu", .3, "isotropic friction coefficient");
    cli.AddDouble("mu_static", -1, "optional distinct static friction");
    cli.AddDouble("eps_velocity", .001, "friction regularization speed");
    cli.AddDouble("slope", 10, "slope degrees for brick");
    cli.AddDouble("gap", .001, "initial surface gap");
    cli.AddString("material", "svk", "svk | mr");
    cli.AddString("integrator", "be", "be | bdf2");
    cli.AddBool("self_collision", false, "enable within-body contact");
    cli.AddInt("export_interval", 0, "VTU interval, zero disables");
    cli.AddString("out", "output/ipc", "output directory");
    std::string error;
    if (!cli.Parse(argc, argv, &error)) {
      std::cerr << error << '\n';
      return 2;
    }
    if (cli.HelpRequested()) {
      cli.PrintUsage(std::cout);
      return 0;
    }
    const auto scenario = cli.GetString("case");
    const int res = cli.GetInt("resolution"), steps = cli.GetInt("steps");
    if (res < 1 || res > 20 || steps < 1 ||
        (scenario != "brick" && scenario != "sphere" && scenario != "two-body"))
      throw std::invalid_argument("Invalid demo case, resolution or steps");
    const double gap = cli.GetDouble("gap");
    if (!(gap > cli.GetDouble("dmin")))
      throw std::invalid_argument("Initial gap must exceed dmin");
    std::vector<Mesh> meshes;
    if (scenario == "two-body") {
      meshes.push_back(Box({-.05, -.05, -.1}, {.05, .05, 0}, res));
      meshes.push_back(Box({-.05, -.05, gap}, {.05, .05, .1 + gap}, res));
    } else {
      meshes.push_back(Box({-.3, -.3, -.1}, {.3, .3, 0}, res));
      if (scenario == "sphere") {
        ANCFCPUUtils::MeshManager manager;
        const std::string sphere_res = cli.GetString("sphere_resolution");
        if (sphere_res != "low" && sphere_res != "med" && sphere_res != "high")
          throw std::invalid_argument("Invalid sphere resolution");
        const std::string prefix =
            "data/meshes/ptest/sphere/sphere_r0p15_" + sphere_res + ".1";
        int id = manager.LoadMesh(prefix + ".node", prefix + ".ele", "sphere");
        if (id < 0)
          throw std::runtime_error("Could not load sphere mesh");
        Mesh sphere{manager.GetAllNodes(), manager.GetAllElements()};
        sphere.X.col(0).array() -=
            .5 * (sphere.X.col(0).maxCoeff() + sphere.X.col(0).minCoeff());
        sphere.X.col(1).array() -=
            .5 * (sphere.X.col(1).maxCoeff() + sphere.X.col(1).minCoeff());
        sphere.X.col(2).array() += gap - sphere.X.col(2).minCoeff();
        meshes.push_back(std::move(sphere));
      } else
        meshes.push_back(Box({-.05, -.05, gap}, {.05, .05, .1 + gap}, res));
    }
    std::vector<int> fixed;
    if (scenario != "two-body")
      for (int i = 0; i < meshes[0].X.rows(); ++i)
        if (std::abs(meshes[0].X(i, 2) + .1) < 1e-12)
          fixed.push_back(i);
    if (scenario == "brick") {
      const Eigen::Matrix3d R =
          Eigen::AngleAxisd(cli.GetDouble("slope") * std::acos(-1.) / 180,
                            Eigen::Vector3d::UnitY())
              .toRotationMatrix();
      for (auto& m : meshes)
        m.X = m.X * R.transpose();
    }
    int nn = 0, ne = 0;
    for (const auto& m : meshes) {
      nn += m.X.rows();
      ne += m.T.rows();
    }
    Eigen::MatrixXd X(nn, 3);
    Eigen::MatrixXi T(ne, 10);
    std::vector<ANCFCPUUtils::MeshInstance> instances;
    int nv = 0, nt = 0;
    for (const auto& m : meshes) {
      instances.push_back({nv, nt, int(m.X.rows()), int(m.T.rows()), "body"});
      X.middleRows(nv, m.X.rows()) = m.X;
      T.middleRows(nt, m.T.rows()) = (m.T.array() + nv).matrix();
      nv += m.X.rows();
      nt += m.T.rows();
    }
    std::cout << "Preparing nodes=" << nn << " elements=" << ne << std::endl;
    GPU_FEAT10_Data data(ne, nn);
    data.Initialize();
    data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
               Quadrature::tet5pt_weights, X.col(0), X.col(1), X.col(2), T);
    const double E = cli.GetDouble("E"), density = cli.GetDouble("density");
    const std::string material = cli.GetString("material");
    if (material == "svk")
      data.ApplyMaterial(SolidMaterialProperties::SVK(E, .3, density, 0, 0));
    else if (material == "mr")
      data.ApplyMaterial(SolidMaterialProperties::MooneyRivlin(
          E / 6, 0, E / (3 * (1 - 2 * .3)), density, 0, 0));
    else
      throw std::invalid_argument("Unknown material");
    if (!fixed.empty())
      data.SetNodalFixed(
          Eigen::Map<Eigen::VectorXi>(fixed.data(), fixed.size()));
    std::vector<ANCFCPUUtils::SurfaceTriMesh> surfaces;
    for (const auto& i : instances)
      surfaces.push_back(ANCFCPUUtils::ExtractSurfaceTriMesh(X, T, i));
    contact::Config c;
    c.dhat           = cli.GetDouble("dhat");
    c.dmin           = cli.GetDouble("dmin");
    c.stiffness      = cli.GetDouble("stiffness");
    c.mu_kinetic     = cli.GetDouble("mu");
    c.mu_static      = cli.GetDouble("mu_static") < 0 ? c.mu_kinetic
                                                      : cli.GetDouble("mu_static");
    c.eps_velocity   = cli.GetDouble("eps_velocity");
    c.self_collision = cli.GetBool("self_collision");
    auto contact     = std::make_unique<contact::IpcContact>(
        contact::Surface::FromMeshes(X, surfaces), c);
    std::cout << "Assembling FE reference data" << std::endl;
    IpcT10Model model(data);
    Eigen::VectorXd gravity = Eigen::VectorXd::Zero(3 * nn);
    if (scenario != "two-body")
      for (int n = 0; n < nn; ++n)
        gravity[3 * n + 2] = -9.81;
    data.SetExternalForce(model.mass() * gravity);
    const Eigen::VectorXd mass_weights =
        model.mass() * Eigen::VectorXd::Ones(3 * nn);
    double body_mass = 0, body_reference_z = 0;
    for (int i = instances[1].node_offset; i < nn; ++i) {
      body_mass += mass_weights[3 * i + 2];
      body_reference_z += mass_weights[3 * i + 2] * X(i, 2);
    }
    body_reference_z /= body_mass;

    IpcNewtonParams params;
    params.max_newton           = cli.GetInt("max_newton");
    params.max_outer            = cli.GetInt("max_outer");
    params.max_retries          = cli.GetInt("max_retries");
    params.verbose              = cli.GetBool("verbose");
    params.time_step            = cli.GetDouble("dt");
    params.min_time_step        = params.time_step / 1024;
    params.gradient_atol        = 1e-6;
    params.constraint_tolerance = 1e-7;
    const auto integrator       = cli.GetString("integrator");
    if (integrator == "bdf2")
      params.integrator = IpcIntegrator::BDF2;
    else if (integrator != "be")
      throw std::invalid_argument("Unknown integrator");
    std::cout << "Initializing coupled solver" << std::endl;
    IpcNewtonSolver solver(data, std::move(contact), params);
    if (scenario == "two-body") {
      Eigen::MatrixXd v = Eigen::MatrixXd::Zero(nn, 3);
      v.topRows(meshes[0].X.rows()).col(2).setConstant(.05);
      v.bottomRows(meshes[1].X.rows()).col(2).setConstant(-.05);
      solver.SetInitialVelocity(Flatten(v));
    }
    const std::string out = cli.GetString("out");
    std::filesystem::create_directories(out);
    std::ofstream csv(out + "/steps.csv");
    csv << "step,time,dt,retries,newton,outer,contacts,nnz,analyses,residual,"
           "constraint,min_J,min_alpha,energy,seconds,kinetic,px,py,pz,elastic_"
           "seconds,contact_seconds,ccd_seconds,linear_seconds,assembly_"
           "seconds,active_min_distance,body_z,body_contact_z,"
           "body_displacement_z,body_mass\n";
    csv << std::setprecision(17);
    std::ofstream config(out + "/config.txt");
    for (int i = 0; i < argc; ++i)
      config << argv[i] << ' ';
    config << "\nToolkit=1.6.1 CPU; mass and elastic quadrature=125 positive "
              "Duffy points; damping=0\n";
    std::cout << "nodes=" << nn << " elements=" << ne << " case=" << scenario
              << '\n';
    for (int step = 0; step < steps; ++step) {
      const auto r = solver.Advance(params.time_step);
      std::cout << "step=" << step << " time=" << r.time
                << " accepted=" << r.accepted
                << " newton=" << r.newton_iterations
                << " contacts=" << r.contacts << " retries=" << r.retries
                << " reason=" << r.reason << std::endl;
      for (const auto& failure : r.failed_attempts)
        std::cout << "  retry: " << failure << std::endl;
      if (!r.accepted) {
        data.Destroy();
        return 1;
      }
      const auto v             = solver.Velocity();
      const Eigen::VectorXd mv = model.mass() * v;
      Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
      for (int n = 0; n < nn; ++n)
        momentum += mv.segment<3>(3 * n);
      const auto positions = model.Positions();
      double body_z        = 0;
      for (int i = instances[1].node_offset; i < nn; ++i)
        body_z += mass_weights[3 * i + 2] * positions(i, 2);
      body_z /= body_mass;
      csv << step << ',' << r.time << ',' << r.step_size << ',' << r.retries
          << ',' << r.newton_iterations << ',' << r.outer_iterations << ','
          << r.contacts << ',' << r.matrix_nonzeros << ','
          << r.symbolic_analyses << ',' << r.residual << ','
          << r.constraint_error << ',' << r.min_jacobian << ',' << r.min_alpha
          << ',' << r.energy << ',' << r.elapsed_seconds << ','
          << .5 * v.dot(mv) << ',' << momentum[0] << ',' << momentum[1] << ','
          << momentum[2] << ',' << r.elastic_seconds << ',' << r.contact_seconds
          << ',' << r.ccd_seconds << ',' << r.linear_seconds << ','
          << r.assembly_seconds << ',' << r.active_min_distance << ',' << body_z
          << ','
          << (r.contact_force.size()
                  ? Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                                   Eigen::RowMajor>>(
                        r.contact_force.data(), nn, 3)
                        .bottomRows(meshes[1].X.rows())
                        .col(2)
                        .sum()
                  : 0)
          << ',' << body_z - body_reference_z << ',' << body_mass << '\n';
      csv.flush();
      const int interval = cli.GetInt("export_interval");
      if (interval > 0 && step % interval == 0)
        data.WriteOutputVTU(out + "/step_" + std::to_string(step) + ".vtu");
    }
    data.Destroy();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
