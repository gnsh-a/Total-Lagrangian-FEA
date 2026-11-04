#include "IpcContact.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <ipc/ccd/tight_inclusion_ccd.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>
#include <ipc/ipc.hpp>
#include <ipc/potentials/barrier_potential.hpp>
#include <ipc/potentials/friction_potential.hpp>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace tlfea::contact {
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::invalid_argument(message);
}
using RowSparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
}  // namespace
Surface Surface::FromMeshes(
    const Eigen::MatrixXd& X,
    const std::vector<ANCFCPUUtils::SurfaceTriMesh>& meshes) {
  Surface s;
  s.full_rest = X;
  size_t nv = 0, nf = 0;
  for (const auto& m : meshes) {
    nv += m.vertices.size();
    nf += m.triangles.size();
  }
  Require(nv < INT_MAX / 3 && nf < INT_MAX,
          "IPC surface exceeds 32-bit indexing");
  s.rest.resize(nv, 3);
  s.faces.resize(nf, 3);
  s.W.resize(nv, X.rows());
  std::vector<Eigen::Triplet<double>> w;
  int v = 0, f = 0, b = 0;
  for (const auto& m : meshes) {
    Require(m.ancf_node_ids.empty(),
            "IPC T10 mapping does not accept ANCF synthetic nodes");
    Require(m.global_node_ids.size() == m.vertices.size(),
            "Missing surface node map");
    for (size_t i = 0; i < m.vertices.size(); ++i) {
      const int n = m.global_node_ids[i];
      Require(n >= 0 && n < X.rows(), "Invalid surface node ID");
      s.rest.row(v + i) = m.vertices[i];
      w.emplace_back(v + i, n, 1);
      s.body.push_back(b);
      s.kinematic.push_back(false);
    }
    for (const auto& t : m.triangles)
      s.faces.row(f++) = (t.array() + v).matrix();
    v += m.vertices.size();
    ++b;
  }
  s.W.setFromTriplets(w.begin(), w.end());
  return s;
}
struct IpcContact::Impl {
  Surface s;
  Config cfg;
  RowSparse W;
  ipc::CollisionMesh mesh;
  ipc::BarrierPotential barrier;
  ipc::FrictionPotential friction;
  ipc::TangentialCollisions lag;
  ipc::TightInclusionCCD ccd;
  struct BodyBoundary {
    std::vector<int> faces, representatives;
    bool closed = true;
  };
  std::map<int, BodyBoundary> boundaries;
  Impl(Surface surface, Config c)
      : s(std::move(surface)),
        cfg(c),
        W(s.W),
        barrier(c.dhat, c.stiffness, true),
        friction(c.eps_velocity),
        ccd(c.ccd_tolerance) {
    Require(std::isfinite(c.dhat) && c.dhat > 0 && std::isfinite(c.dmin) &&
                c.dmin >= 0,
            "IPC requires dhat>0 and dmin>=0");
    Require(std::isfinite(c.stiffness) && c.stiffness > 0 &&
                std::isfinite(c.eps_velocity) && c.eps_velocity > 0,
            "IPC requires positive stiffness and friction regularization");
    Require(std::isfinite(c.mu_static) && std::isfinite(c.mu_kinetic) &&
                c.mu_static >= c.mu_kinetic && c.mu_kinetic >= 0,
            "IPC requires mu_static>=mu_kinetic>=0");
    Require(std::isfinite(c.ccd_tolerance) && c.ccd_tolerance > 0,
            "Invalid CCD tolerance");
    Require(s.rest.cols() == 3 && s.full_rest.cols() == 3 &&
                s.rest.allFinite() && s.full_rest.allFinite(),
            "Invalid IPC rest positions");
    Require(s.W.rows() == s.rest.rows() && s.W.cols() == s.full_rest.rows(),
            "Invalid IPC displacement map dimensions");
    Require(s.body.size() == size_t(s.rest.rows()), "Missing IPC body IDs");
    Require(s.rest.rows() < INT_MAX / 3 && s.full_rest.rows() < INT_MAX / 3,
            "IPC mesh exceeds 32-bit DOF indexing");
    if (s.kinematic.empty())
      s.kinematic.assign(s.rest.rows(), false);
    Require(s.kinematic.size() == s.body.size(), "Invalid IPC kinematic mask");
    for (int i = 0; i < W.rows(); ++i) {
      double sum   = 0;
      bool nonzero = false;
      for (RowSparse::InnerIterator it(W, i); it; ++it) {
        Require(std::isfinite(it.value()), "Nonfinite IPC map");
        sum += it.value();
        nonzero = nonzero || it.value() != 0;
      }
      Require(s.kinematic[i] ? !nonzero : std::abs(sum - 1) < 1e-12,
              "Dynamic IPC map rows must preserve translations; kinematic rows "
              "must be empty");
    }
    std::set<std::pair<int, int>> edges;
    std::set<std::array<int, 3>> faces;
    Require(s.faces.cols() == 3, "IPC faces must be triangles");
    for (int i = 0; i < s.faces.rows(); ++i) {
      std::array<int, 3> key;
      for (int j = 0; j < 3; ++j) {
        int a = s.faces(i, j), b = s.faces(i, (j + 1) % 3);
        Require(a >= 0 && a < s.rest.rows() && b >= 0 && b < s.rest.rows() &&
                    a != b,
                "Invalid IPC triangle");
        Require(s.body[a] == s.body[b], "IPC triangle spans bodies");
        edges.emplace(std::min(a, b), std::max(a, b));
        key[j] = a;
      }
      std::sort(key.begin(), key.end());
      Require(faces.insert(key).second, "Duplicate IPC face");
      const Eigen::Vector3d a =
          s.rest.row(s.faces(i, 1)) - s.rest.row(s.faces(i, 0));
      const Eigen::Vector3d b =
          s.rest.row(s.faces(i, 2)) - s.rest.row(s.faces(i, 0));
      Require(a.cross(b).squaredNorm() > 0, "Degenerate IPC triangle");
    }
    Eigen::MatrixXi E(edges.size(), 2);
    int i = 0;
    for (auto e : edges) {
      E(i, 0)   = e.first;
      E(i++, 1) = e.second;
    }
    mesh             = ipc::CollisionMesh(s.rest, E, s.faces);
    mesh.can_collide = [this](size_t a, size_t b) {
      if (s.kinematic[a] && s.kinematic[b])
        return false;
      if (!cfg.self_collision && s.body[a] == s.body[b])
        return false;
      for (auto p : s.excluded_body_pairs)
        if ((s.body[a] == p.first && s.body[b] == p.second) ||
            (s.body[a] == p.second && s.body[b] == p.first))
          return false;
      return true;
    };
    // Surface intersection tests alone miss one solid wholly inside another.
    // Keep connected-component samples and closed body boundaries for a
    // winding-number containment check at the initial state.
    std::vector<int> parent(s.rest.rows());
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int v) {
      while (parent[v] != v) {
        parent[v] = parent[parent[v]];
        v         = parent[v];
      }
      return v;
    };
    std::map<std::pair<int, int>, std::pair<int, int>> incidence;
    for (int f = 0; f < s.faces.rows(); ++f) {
      boundaries[s.body[s.faces(f, 0)]].faces.push_back(f);
      for (int j = 0; j < 3; ++j) {
        int a = s.faces(f, j), b = s.faces(f, (j + 1) % 3);
        parent[root(a)] = root(b);
        auto& count     = incidence[std::minmax(a, b)];
        ++count.first;
        count.second += a < b ? 1 : -1;
      }
    }
    for (const auto& entry : incidence) {
      auto& body = boundaries[s.body[entry.first.first]];
      body.closed =
          body.closed && entry.second.first == 2 && entry.second.second == 0;
    }
    std::set<int> seen;
    for (int f = 0; f < s.faces.rows(); ++f) {
      const int v = s.faces(f, 0), r = root(v);
      if (seen.insert(r).second)
        boundaries[s.body[v]].representatives.push_back(v);
    }
  }
  void ValidateContainment(const Eigen::MatrixXd& V) const {
    for (const auto& target : boundaries) {
      const auto& body = target.second;
      if (!body.closed || body.faces.empty())
        continue;
      Eigen::Vector3d low = Eigen::Vector3d::Constant(INFINITY), high = -low;
      for (int f : body.faces)
        for (int j = 0; j < 3; ++j) {
          low  = low.cwiseMin(V.row(s.faces(f, j)).transpose());
          high = high.cwiseMax(V.row(s.faces(f, j)).transpose());
        }
      const int other = s.faces(body.faces.front(), 0);
      for (const auto& source : boundaries) {
        if (source.first == target.first)
          continue;
        for (int vertex : source.second.representatives) {
          if (!mesh.can_collide(vertex, other))
            continue;
          const Eigen::Vector3d p = V.row(vertex);
          if ((p.array() < low.array()).any() ||
              (p.array() > high.array()).any())
            continue;
          double angle = 0;
          for (int f : body.faces) {
            const Eigen::Vector3d a = V.row(s.faces(f, 0)).transpose() - p,
                                  b = V.row(s.faces(f, 1)).transpose() - p,
                                  c = V.row(s.faces(f, 2)).transpose() - p;
            angle +=
                2 * std::atan2(a.dot(b.cross(c)),
                               a.norm() * b.norm() * c.norm() +
                                   a.dot(b) * c.norm() + b.dot(c) * a.norm() +
                                   c.dot(a) * b.norm());
          }
          Require(std::abs(angle) < 2 * std::acos(-1.),
                  "IPC initial solids overlap by containment");
        }
      }
    }
  }
  ipc::NormalCollisions Collisions(const Eigen::MatrixXd& V) const {
    ipc::NormalCollisions n;
    n.set_use_area_weighting(cfg.area_weighting);
    n.set_collision_set_type(
        ipc::NormalCollisions::CollisionSetType::IMPROVED_MAX_APPROX);
    n.build(mesh, V, cfg.dhat, cfg.dmin);
    return n;
  }
  template <class Set, class Potential>
  Evaluation Evaluate(const Set& set, const Potential& p,
                      const Eigen::MatrixXd& V, bool derivatives,
                      bool psd) const {
    Evaluation r;
    r.collisions = set.size();
    r.energy     = p(set, mesh, V);
    if (!derivatives)
      return r;
    r.gradient = Eigen::VectorXd::Zero(3 * s.full_rest.rows());
    r.hessian.resize(r.gradient.size(), r.gradient.size());
    std::vector<Eigen::Triplet<double>> entries;
    r.blocks.reserve(set.size());
    for (size_t k = 0; k < set.size(); ++k) {
      const auto& c  = set[k];
      const auto ids = c.vertex_ids(mesh.edges(), mesh.faces());
      const auto x   = c.dof(V, mesh.edges(), mesh.faces());
      LocalBlock block;
      block.vertices.assign(ids.begin(), ids.begin() + c.num_vertices());
      block.gradient = p.gradient(c, x);
      block.hessian  = p.hessian(c, x,
                                psd ? ipc::PSDProjectionMethod::CLAMP
                                     : ipc::PSDProjectionMethod::NONE);
      for (int i = 0; i < c.num_vertices(); ++i)
        for (RowSparse::InnerIterator a(W, ids[i]); a; ++a) {
          r.gradient.segment<3>(3 * a.col()) +=
              a.value() * block.gradient.segment<3>(3 * i);
          for (int j = 0; j < c.num_vertices(); ++j)
            for (RowSparse::InnerIterator b(W, ids[j]); b; ++b)
              for (int d = 0; d < 3; ++d)
                for (int e = 0; e < 3; ++e)
                  entries.emplace_back(3 * a.col() + d, 3 * b.col() + e,
                                       a.value() * b.value() *
                                           block.hessian(3 * i + d, 3 * j + e));
        }
      r.blocks.push_back(std::move(block));
    }
    r.hessian.setFromTriplets(entries.begin(), entries.end());
    return r;
  }
};
IpcContact::IpcContact(Surface s, Config c)
    : impl_(std::make_unique<Impl>(std::move(s), c)) {}
IpcContact::~IpcContact() = default;
const Config& IpcContact::config() const {
  return impl_->cfg;
}
const Surface& IpcContact::surface() const {
  return impl_->s;
}
Eigen::MatrixXd IpcContact::Positions(const Eigen::MatrixXd& x) const {
  Require(
      x.rows() == impl_->s.full_rest.rows() && x.cols() == 3 && x.allFinite(),
      "Invalid IPC positions");
  return impl_->s.rest + impl_->W * (x - impl_->s.full_rest);
}
void IpcContact::ValidateInitial(const Eigen::MatrixXd& x) const {
  const auto V = Positions(x);
  Require(!ipc::has_intersections(impl_->mesh, V),
          "IPC initial surface is intersecting");
  impl_->ValidateContainment(V);
  const auto n = impl_->Collisions(V);
  for (size_t i = 0; i < n.size(); ++i)
    Require(n[i].compute_distance(V, impl_->mesh.edges(), impl_->mesh.faces()) >
                impl_->cfg.dmin * impl_->cfg.dmin,
            "IPC initial surface violates minimum separation");
}
double IpcContact::StepSize(const Eigen::MatrixXd& from,
                            const Eigen::MatrixXd& to) const {
  const double a = ipc::compute_collision_free_stepsize(
      impl_->mesh, Positions(from), Positions(to), impl_->cfg.dmin, nullptr,
      impl_->ccd);
  if (!std::isfinite(a) || a < 0 || a > 1)
    throw std::runtime_error("Invalid IPC CCD result");
  return a;
}
Evaluation IpcContact::Normal(const Eigen::MatrixXd& x, bool derivatives,
                              bool psd) const {
  const auto V = Positions(x);
  const auto n = impl_->Collisions(V);
  auto result  = impl_->Evaluate(n, impl_->barrier, V, derivatives, psd);
  result.active_min_distance =
      std::sqrt(n.compute_minimum_distance(impl_->mesh, V));
  return result;
}
Evaluation IpcContact::Friction(const Eigen::MatrixXd& v, bool derivatives,
                                bool psd) const {
  Require(
      v.rows() == impl_->s.full_rest.rows() && v.cols() == 3 && v.allFinite(),
      "Invalid IPC velocity");
  return impl_->Evaluate(impl_->lag, impl_->friction,
                         Eigen::MatrixXd(impl_->W * v), derivatives, psd);
}
void IpcContact::RefreshFriction(const Eigen::MatrixXd& x) {
  impl_->lag.clear();
  if (impl_->cfg.mu_static == 0)
    return;
  const auto V = Positions(x);
  const auto n = impl_->Collisions(V);
  impl_->lag.build(impl_->mesh, V, n, impl_->barrier, impl_->cfg.mu_static,
                   impl_->cfg.mu_kinetic);
}
IpcContact::Checkpoint IpcContact::Save() const {
  return {std::make_shared<ipc::TangentialCollisions>(impl_->lag)};
}
void IpcContact::Restore(const Checkpoint& s) {
  Require(bool(s.state), "Missing IPC checkpoint");
  impl_->lag =
      *std::static_pointer_cast<const ipc::TangentialCollisions>(s.state);
}
}  // namespace tlfea::contact
