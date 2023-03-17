#ifndef DEF_CubicLagrangeMinimize
#define DEF_CubicLagrangeMinimize

#include <cmath>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix4d;
using Eigen::Vector4d;

/// @brief Function to find minimum of f over interval [a, b] using cubic Lagrange polynomial interpolation.
/// If the function is monotonic, then the minimum is one of the bounds of the interval, and the minimum is found in a single iteration.
/// The best estimate of the minimum within the current interval is returned once the interval is smaller than the tolerance.
/// The number of function evaluations is 2 + 2*Niter.
/// The interval width is reduced by a factor of 3 every iteration.
/// @param f The univariate function to minimize.
/// @param a The lower bound of the interval.
/// @param b The upper bound of the interval.
/// @param tol The tolerance on the interval width.
/// @return The best estimate of the minimum within the current interval once its width is smaller than the tolerance.
double CubicLagrangeMinimize(std::function<double(double)> f, double a, double b, double tol=1e-9);

#endif