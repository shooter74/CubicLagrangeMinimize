#ifndef DEF_CubicLagrangeMinimize
#define DEF_CubicLagrangeMinimize

#include <cmath>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix4d;
using Eigen::Vector4d;

/// @brief Returns a callback function that does nothing.
#define CubicLagrangeMinimize_GetEmptyCallback() [](Eigen::VectorXd const&){}

/// @brief Returns a callback function that prints the current best estimate of the minimum and the iteration number.
#define CubicLagrangeMinimize_GetSimpleCallback() [](Eigen::VectorXd const& callback_args){ std::cout << "Iteration " << callback_args[2] << " : f(" << callback_args[0] << ") = " << callback_args[1] << "\n"; }

/// @brief Returns a callback function that prints the current best estimate of the minimum, the iteration number, the current interval, and the cubic polynomial coefficients.
#define CubicLagrangeMinimize_GetDetailedCallback() \
[](Eigen::VectorXd const& callback_args){ \
    std::cout << "--------------------------------------------------------------------\n";\
    std::cout << "                            Iteration " << callback_args[2] << "\n";\
    std::cout << "--------------------------------------------------------------------\n";\
    std::cout << "X values : " << callback_args[8] << ", " << callback_args[9] << ", " << callback_args[10] << ", " << callback_args[11] << "\n";\
    std::cout << "Cubic poly coeffs : " << callback_args[12] << ", " << callback_args[13] << ", " << callback_args[14] << ", " << callback_args[15] << "\n";\
    std::cout << "Quadratic solution : delta = " << callback_args[3] << ", x_sol_1 = " << callback_args[4] << ", x_sol_2 = " << callback_args[5] << ", y_sol_1 = " << callback_args[6] << ", y_sol_2 = " << callback_args[7] << "\n";\
    std::cout << "Current solution : f(" << callback_args[0] << ") = " << callback_args[1] << "\n";\
}

/// @brief Function to find minimum of f over interval [a, b] using cubic Lagrange polynomial interpolation.
/// If the function is monotonic, then the minimum is one of the bounds of the interval, and the minimum is found in a single iteration.
/// The best estimate of the minimum within the current interval is returned once the interval is smaller than the tolerance.
/// The number of function evaluations is 2 + 2*Niter.
/// The interval width is reduced by a factor of 3 every iteration.
/// @param f The univariate function to minimize.
/// @param a The lower bound of the interval.
/// @param b The upper bound of the interval.
/// @param tol The tolerance on the interval width.
/// @param callback A function to call at each iteration with the current best estimate of the minimum and other internal variables. The vector passed to the callback function contains the following variables : (x_sol, y_sol, i, delta, x_sol_1, x_sol_2, y_sol_1, y_sol_2, x0, x1, x2, x3, a0, a1, a2, a3).
/// @return The best estimate of the minimum within the current interval once its width is smaller than the tolerance.
double CubicLagrangeMinimize(std::function<double(double)> f, double a, double b, double tol=1e-9, std::function<void(Eigen::VectorXd const&)> callback = [](Eigen::VectorXd const&){});

// (x_sol, y_sol, i, delta, x_sol_1, x_sol_2, y_sol_1, y_sol_2, x0, x1, x2, x3, a0, a1, a2, a3).

#endif