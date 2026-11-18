// Rigid brick on a rigid slope, Chrono SMC (penalty) contact.
//
// Same brick as the TL-FEA / IPC / DEME brick tests: 0.2 x 0.1 x 0.05 m, 1 kg, starting at rest in contact.
// The slope is a flat fixed box with gravity tilted by the slope angle (x = down the slope, z = slope normal).
// Writes t, s (distance down the slope), normal offset, velocity and contact force to CSV and prints the
// slid distance against the rigid-body analytic s = 1/2 g (sin a - mu cos a) t^2 (0 if tan a <= mu).
//
//   brick_slope_smc --slope 04 --mu 0.2 --dt 1e-4 --model hertz --out out/s04.csv

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include "chrono/collision/ChCollisionInfo.h"
#include "chrono/collision/ChCollisionModel.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChSystemSMC.h"

using namespace chrono;

static const std::map<std::string, double> kSlope = {
    {"01", 0.18},         {"02", 0.197395560},  {"03", 0.244978663},          {"03p1", 0.2440372654},
    {"03p2", 0.2459196179}, {"03p3", std::atan(0.247)}, {"03p4", std::atan(0.248)}, {"04", 0.25}};

int main(int argc, char* argv[]) {
    std::string slope = "04", model = "hertz", out = "";
    double mu = 0.2, dt = 1e-4, t_end = 0.3, E = 1e7, nu = 0.3, cr = 0.1;
    bool stiff = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--slope") slope = v;
        else if (k == "--mu") mu = std::atof(v.c_str());
        else if (k == "--dt") dt = std::atof(v.c_str());
        else if (k == "--t_end") t_end = std::atof(v.c_str());
        else if (k == "--E") E = std::atof(v.c_str());
        else if (k == "--cr") cr = std::atof(v.c_str());
        else if (k == "--model") model = v;
        else if (k == "--stiff") stiff = std::atoi(v.c_str()) != 0;
        else if (k == "--out") out = v;
        else { std::cerr << "unknown option " << k << "\n"; return 1; }
    }
    if (!kSlope.count(slope)) { std::cerr << "unknown slope " << slope << "\n"; return 1; }
    if (!(dt > 0) || !(t_end > 0) || !(mu >= 0)) { std::cerr << "need dt > 0, t_end > 0, mu >= 0\n"; return 1; }
    const double alpha = kSlope.at(slope), g = 9.81;
    const double lx = 0.2, ly = 0.1, lz = 0.05, mass = 1.0;

    ChSystemSMC sys;
    sys.SetCollisionSystemType(ChCollisionSystem::Type::BULLET);
    sys.SetGravitationalAcceleration(ChVector3d(g * std::sin(alpha), 0, -g * std::cos(alpha)));
    std::map<std::string, ChSystemSMC::ContactForceModel> models = {
        {"hooke", ChSystemSMC::Hooke}, {"hertz", ChSystemSMC::Hertz},
        {"plaincoulomb", ChSystemSMC::PlainCoulomb}, {"flores", ChSystemSMC::Flores}};
    if (!models.count(model)) { std::cerr << "unknown model " << model << "\n"; return 1; }
    sys.SetContactForceModel(models.at(model));
    sys.SetTangentialDisplacementModel(ChSystemSMC::OneStep);  // MultiStep falls back to OneStep in Chrono 10
    sys.SetContactStiff(stiff);
    ChCollisionModel::SetDefaultSuggestedEnvelope(1e-3);
    ChCollisionModel::SetDefaultSuggestedMargin(1e-3);

    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();
    mat->SetYoungModulus(static_cast<float>(E));
    mat->SetPoissonRatio(static_cast<float>(nu));
    mat->SetRestitution(static_cast<float>(cr));
    mat->SetFriction(static_cast<float>(mu));  // SMC uses one coefficient (the static one) for stick and slip

    auto ground = chrono_types::make_shared<ChBodyEasyBox>(3.0, 1.0, 0.1, 1000, false, true, mat);
    ground->SetPos(ChVector3d(0.5, 0, -0.05));  // top face at z = 0
    ground->SetFixed(true);
    sys.AddBody(ground);

    auto brick = chrono_types::make_shared<ChBodyEasyBox>(lx, ly, lz, mass / (lx * ly * lz), false, true, mat);
    const ChVector3d p0(0, 0, 0.5 * lz);  // resting on the slope, zero gap
    brick->SetPos(p0);
    sys.AddBody(brick);

    std::ofstream csv;
    if (!out.empty()) {
        csv.open(out);
        csv << "t,s,z,vx,vz,fx,fz,ncontacts\n";
    }
    const int n_steps = static_cast<int>(std::lround(t_end / dt));
    const int every = std::max(1, static_cast<int>(std::lround(5e-4 / dt)));  // CSV row every 0.5 ms
    for (int k = 0; k <= n_steps; ++k) {
        if (k > 0) sys.DoStepDynamics(dt);
        if (csv.is_open() && k % every == 0) {
            const ChVector3d p = brick->GetPos() - p0, v = brick->GetPosDt(), f = brick->GetContactForce();
            csv << std::setprecision(10) << sys.GetChTime() << "," << p.x() << "," << p.z() << "," << v.x() << ","
                << v.z() << "," << f.x() << "," << f.z() << "," << sys.GetNumContacts() << "\n";
        }
    }
    const double t = sys.GetChTime();
    const double a = std::tan(alpha) <= mu ? 0.0 : g * (std::sin(alpha) - mu * std::cos(alpha));
    const double s = brick->GetPos().x() - p0.x(), s_ref = 0.5 * a * t * t;
    std::cout << std::fixed << std::setprecision(3) << "slope " << slope << " (" << alpha * 180 / CH_PI
              << " deg, tan " << std::tan(alpha) << ")  mu " << mu << "  model " << model << "  dt " << std::scientific << std::setprecision(1) << dt << std::fixed << std::setprecision(3)
              << "  stiff " << stiff << "  t " << t << " s\n"
              << "  slid " << s * 1e3 << " mm  (analytic " << s_ref * 1e3 << " mm, "
              << (a > 0 ? "slide" : "stick") << ")  normal offset " << (brick->GetPos().z() - p0.z()) * 1e6
              << " um  contacts " << sys.GetNumContacts() << "\n";
    return 0;
}
