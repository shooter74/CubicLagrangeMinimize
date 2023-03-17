#include <iostream>
#include <vector>
#include <CubicLagrangeMinimize.hpp>

using std::cout;
using std::endl;

double fct_01(double x) {
    return x*x + std::sin(5.*x);
}

double dfct_01(double x) {
    return 2.*x + 5.*std::cos(5.*x);
}

double fct_02(double x) {
    return std::exp(x);
}

double dfct_02(double x) {
    return std::exp(x);
}

double fct_03(double x) {
    return -std::exp(-x*x);
}

double dfct_03(double x) {
    return 2.*x*std::exp(-x*x);
}

double fct_04(double x) {
    return std::exp(-x);
}

double dfct_04(double x) {
    return std::exp(-x);
}

double fct_05(double x) {
    return std::sin(x);
}

double dfct_05(double x) {
    return std::cos(x);
}

double fct_06(double x) {
    return x*x;
}

double dfct_06(double x) {
    return 2.*x;
}

double fct_07(double x) {
    return x;
}

double dfct_07(double) {
    return 1.;
}

void test_CubicLagrangeMinimize() {
    std::vector<std::function<double(double)>> fcts = {fct_01, fct_02, fct_03, fct_04, fct_05, fct_06, fct_07};
    std::vector<std::function<double(double)>> dfcts = {dfct_01, dfct_02, dfct_03, dfct_04, dfct_05, dfct_06, dfct_07};
    std::vector<double> mins = {-1.2, -10., -1.5, -10., -1., -2., -3.};
    std::vector<double> maxs = {1.5, 10., 3., 5., 6., 3., 4.};
    for(unsigned int i = 0 ; i < fcts.size() ; ++i) {
        auto f = fcts[i]; auto df = dfcts[i];
        auto x_min = CubicLagrangeMinimize(f, mins[i], maxs[i]);
        cout << "[" << mins[i] << " " << maxs[i] << "]\tf(" << x_min << ") = " << f(x_min) << " df(x_min)/dt = " << df(x_min) << endl;
    }
}

int main(int, char**) {
    std::cout.precision(15);
    test_CubicLagrangeMinimize();
    return 0;
}
