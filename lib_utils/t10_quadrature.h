#pragma once

// Positive tensor-product Gauss rule mapped to the reference tetrahedron.
// Five points per direction integrate Ni*Nj*det(J) exactly for quadratic
// isoparametric geometry (degrees at most 9, 8, 7 after the Duffy map).
// Deliberately independent of the constitutive quadrature rule.
#ifdef __CUDACC__
#define T10_HD __host__ __device__
#else
#define T10_HD
#endif
namespace t10 {
constexpr int kMassPoints = 125;
T10_HD inline void MassPoint(int q, double& x, double& y, double& z,
                             double& w) {
  const double a[5] = {0.0469100770306680036, 0.230765344947158454, 0.5,
                       0.769234655052841546, 0.953089922969331996};
  const double b[5] = {0.118463442528094544, 0.239314335249683234,
                       0.284444444444444444, 0.239314335249683234,
                       0.118463442528094544};
  const int i = q / 25, j = (q / 5) % 5, k = q % 5;
  x = a[i];
  y = (1 - x) * a[j];
  z = (1 - x) * (1 - a[j]) * a[k];
  w = b[i] * b[j] * b[k] * (1 - x) * (1 - x) * (1 - a[j]);
}
T10_HD inline void Shape(double x, double y, double z, double N[10],
                         double G[10][3]) {
  const double L[4]     = {1 - x - y - z, x, y, z};
  const double D[4][3]  = {{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  for (int i = 0; i < 4; ++i) {
    N[i] = L[i] * (2 * L[i] - 1);
    for (int d = 0; d < 3; ++d)
      G[i][d] = (4 * L[i] - 1) * D[i][d];
  }
  for (int e = 0; e < 6; ++e) {
    const int i = edges[e][0], j = edges[e][1];
    N[4 + e] = 4 * L[i] * L[j];
    for (int d = 0; d < 3; ++d)
      G[4 + e][d] = 4 * (L[i] * D[j][d] + L[j] * D[i][d]);
  }
}
T10_HD inline double JacobianDet(const double X[10][3], const double G[10][3]) {
  double J[3][3] = {};
  for (int a = 0; a < 10; ++a)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        J[i][j] += X[a][i] * G[a][j];
  return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
         J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
         J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
}
}  // namespace t10
#undef T10_HD
