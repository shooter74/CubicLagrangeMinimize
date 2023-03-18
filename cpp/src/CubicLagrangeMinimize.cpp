#include <CubicLagrangeMinimize.hpp>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751
#endif

/// @brief Linear least-squares fit of a cubic polynomial to four points.
/// @param x1 Abscissa of first point.
/// @param x2 Abscissa of second point.
/// @param x3 Abscissa of third point.
/// @param x4 Abscissa of fourth point.
/// @param y1 Ordinate of first point.
/// @param y2 Ordinate of second point.
/// @param y3 Ordinate of third point.
/// @param y4 Ordinate of fourth point.
/// @return 4-dimensional vector of polynomial coefficients from low to high order : [a0 a1 a2 a3] -> a0 + a1*x + a2*x^2 + a3*x^3
Vector4d polyfit4(double x1, double x2, double x3, double x4, double y1, double y2, double y3, double y4) {
    Matrix4d A;
    double x1x1 = x1*x1, x2x2 = x2*x2, x3x3 = x3*x3, x4x4 = x4*x4;
    double x1x1x1 = x1x1*x1, x2x2x2 = x2x2*x2, x3x3x3 = x3x3*x3, x4x4x4 = x4x4*x4;
    A << 1, x1, x1x1, x1x1x1,
         1, x2, x2x2, x2x2x2,
         1, x3, x3x3, x3x3x3,
         1, x4, x4x4, x4x4x4;
    VectorXd y(4);
    y << y1, y2, y3, y4;
    return (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * y);
}

/// @brief Linear least-squares fit of a cubic polynomial to four points (-pi/3, y1), (-pi/9, y2), (pi/9, y3), (pi/3, y4).
/// @param y1 Ordinate of first point.
/// @param y2 Ordinate of second point.
/// @param y3 Ordinate of third point.
/// @param y4 Ordinate of fourth point.
/// @return 4-dimensional vector of polynomial coefficients from low to high order : [a0 a1 a2 a3] -> a0 + a1*x + a2*x^2 + a3*x^3
Vector4d polyfit4_mpi3_pi3(double y1, double y2, double y3, double y4) {
    static Matrix4d AT = (Matrix4d() << 1,1,1,1,
                                                -M_PI/3.,-M_PI/9.,M_PI/9.,M_PI/3.,
                                                pow(M_PI,2)/9.,pow(M_PI,2)/81.,pow(M_PI,2)/81.,pow(M_PI,2)/9.,
                                                -pow(M_PI,3)/27.,-pow(M_PI,3)/729.,pow(M_PI,3)/729.,pow(M_PI,3)/27.).finished();// A^T
    static Matrix4d ATAinv = (Matrix4d() << 41./64.,0,-405./(64.*pow(M_PI,2)),0,
                                                    0,3285/(64.*pow(M_PI,2)),0,-29889/(64.*pow(M_PI,4)),
                                                    -405/(64.*pow(M_PI,2)),0,6561/(64.*pow(M_PI,4)),0,
                                                    0,-29889/(64.*pow(M_PI,4)),0,295245/(64.*pow(M_PI,6))).finished();// (A^T * A)^-1
    VectorXd y(4);
    y << y1, y2, y3, y4;
    return ATAinv*(AT*y);
}

/// @brief Cubic polynomial function.
/// @param x    Abscissa.
/// @param a0   Constant term.
/// @param a1   Linear term.
/// @param a2   Quadratic term.
/// @param a3   Cubic term.
/// @return a0 + a1*x + a2*x^2 + a3*x^3
double cubic_poly(double x, double a0, double a1, double a2, double a3) {
    return a0 + a1*x + a2*x*x + a3*x*x*x;
}

/// @brief Maps a value or array "x" from the interval [a1, b1] to the interval [a2, b2].
/// @param x    Value to map.
/// @param a1   Lower bound of the interval to map from.
/// @param b1   Upper bound of the interval to map from.
/// @param a2   Lower bound of the interval to map to.
/// @param b2   Upper bound of the interval to map to.
/// @return Mapped value.
double map_interval(double x, double a1, double b1, double a2, double b2) { return (x - a1) * (b2 - a2) / (b1 - a1) + a2; }

double CubicLagrangeMinimize(std::function<double(double)> f, double a, double b, double tol, std::function<void(Eigen::VectorXd const&)> callback) {
    // initialize interval endpoints and function values
    double x0 = a, x1 = a*2./3. + b*1./3., x2 = a*1./3. + b*2./3., x3 = b;  // endpoints and two points in the interval
    double f0 = f(x0), f1 = 0., f2 = 0., f3 = f(x3);                        // function values at the endpoints and two points in the interval
    double x_prev = (x0 + x3)/2.;                                           // previous value of x_sol to track convergence of the solution
    double a0, a1, a2, a3;                                                  // coefficients of the cubic Lagrange polynomial
    double delta;                                                           // determinant of the quadratic equation of the Lagrange polynomial
    double x_sol, x_sol_1, x_sol_2, y_sol, y_sol_1, y_sol_2;                // solutions of the Lagrange polynomial
    constexpr double small_coefficient = 1e-9;                              // threshold for small coefficients to avoid ill-conditioning of the quadratic equation
    constexpr double x0m = -M_PI/3.;                                        // mapped value of x0 from the interval [x0 x3] to [-pi/3 pi/3]
    constexpr double x3m =  M_PI/3.;                                        // mapped value of x3 from the interval [x0 x3] to [-pi/3 pi/3]

    unsigned int Niter = static_cast<unsigned int>(std::ceil(std::log(std::fabs(b - a)/tol)/std::log(3.)));// number of iterations to reduce interval width by a factor of 3

    for(unsigned int i = 0 ; i < Niter ; ++i) {
        // Compute function values at two points in the interval
        x1 = x0*2./3. + x3*1./3., x2 = x0*1./3. + x3*2./3.;
        f1 = f(x1), f2 = f(x2);

        // compute Lagrange polynomial using least-squares fit to 4 points, which is equivalent to the cubic Lagrange polynomial
        // Use x values remapped to the interval [-pi/3, pi/3] to minimize the condition number of the matrix (A^T * A) in the least-squares fit
        Vector4d A = polyfit4_mpi3_pi3(f0, f1, f2, f3);// Equivalent to Vector4d A = polyfit4(x0, x1, x2, x3, f0, f1, f2, f3);
        a0 = A[0]; a1 = A[1]; a2 = A[2]; a3 = A[3];
        x_sol_1 = 0.; x_sol_2 = 0.; y_sol_1 = 0.; y_sol_2 = 0., delta = 0.;

        // Solve the first derivative of the Lagrange polynomial for a zero
        if(std::fabs(a3) > small_coefficient) {
            delta = -3*a1*a3 + a2*a2;

            if(delta < 0) {
                x_sol = (f0 < f3) ? x0m : x3m;   // just choose the interval tha contains the minimum of the linear polynomial
                y_sol = cubic_poly(x_sol, a0, a1, a2, a3);
            } else {                             // solve for the two solutions of the quadratic equation of the first derivative of the Lagrange polynomial
                x_sol_1 = (-a2 + std::sqrt(delta))/(3.*a3);
                x_sol_2 = (-a2 - std::sqrt(delta))/(3.*a3);

                y_sol_1 = cubic_poly(x_sol_1, a0, a1, a2, a3);
                y_sol_2 = cubic_poly(x_sol_2, a0, a1, a2, a3);

                x_sol = (y_sol_1 < y_sol_2) ? x_sol_1 : x_sol_2;
                y_sol = (y_sol_1 < y_sol_2) ? y_sol_1 : y_sol_2;
            }
        }
        else if(std::fabs(a2) > small_coefficient) { // if a3 is zero, then the Lagrange polynomial is a quadratic polynomial
            x_sol = -a1/(2.*a2);
            y_sol = cubic_poly(x_sol, a0, a1, a2, a3);
        } else {                                     // if a3 and a2 are zero, then the Lagrange polynomial is a linear polynomial
            x_sol = (f0 < f3) ? x0m : x3m;             // just choose the interval tha contains the minimum of the linear polynomial
            y_sol = cubic_poly(x_sol, a0, a1, a2, a3);
        }

        // transform the solution back to the original interval
        x_sol = map_interval(x_sol, x0m, x3m, x0, x3),
        x_sol_1 = map_interval(x_sol_1, x0m, x3m, x0, x3);
        x_sol_2 = map_interval(x_sol_2, x0m, x3m, x0, x3);
        
        // Call the callback function to print progress
        callback((Eigen::VectorXd(16) << x_sol, y_sol, i, delta, x_sol_1, x_sol_2, y_sol_1, y_sol_2, x0, x1, x2, x3, a0, a1, a2, a3).finished());

        // Check convergence
        if(std::fabs(x_sol - x_prev) < tol) { break; }

        // Determine which interval contains the minimum of the cubic polynomial
        if(x_sol < x1) {
            x3 = x1; f3 = f1;
        } else if(x_sol < x2) {
            x0 = x1; f0 = f1;
            x3 = x2; f3 = f2;
        } else {
            x0 = x2; f0 = f2;
        }
        
        x_prev = x_sol;
    }

    // return best estimate of minimum
    if((y_sol < f0) && (y_sol < f3))
        return x_sol;
    else if(f0 < f3)
        return x0;
    else
        return x3;
}
