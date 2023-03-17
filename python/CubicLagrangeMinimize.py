# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cubic_poly(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

def polyfit4(x1, x2, x3, x4, y1, y2, y3, y4):
    A = np.array([[1, x1, x1**2, x1**3], [1, x2, x2**2, x2**3], [1, x3, x3**2, x3**3], [1, x4, x4**2, x4**3]])
    y = np.array([y1, y2, y3, y4])
    a = np.linalg.solve(A.transpose()@A, A.transpose()@y)
    return a

def cubic_lagrange_minimize(f, a, b, tol=1e-6):
    ''' Function to find minimum of f over interval [a, b] using cubic Lagrange polynomial interpolation.
        If the function is monotonic, then the minimum is one of the bounds of the interval, and the minimum is found in a single iteration.
        The best estimate of the minimum is returned.
        The number of function evaluations is 2 + 2*Niter.
    '''
    # initialize interval endpoints and function values
    x0, x1, x2, x3 = a, a*2./3. + b*1./3., a*1./3. + b*2./3., b
    f0, f3 = f(x0), f(x3)
    x_prev = x0
    delta = 1# only needed to debug print
    x_sol_1 = 1# only needed to debug print
    x_sol_2 = 1# only needed to debug print
    y_sol_1 = 1# only needed to debug print
    y_sol_2 = 1# only needed to debug print

    reduction_factor = 3# reduction factor of interval size at each iteration is 3 because we sample two points in the interval each iteration
    Niter = int(np.ceil(np.log(np.abs(b-a)/tol)/np.log(reduction_factor)))

    for i in range(Niter):
        # Compute function values at two points in the interval
        x1, x2 = x0*2./3. + x3*1./3., x0*1./3. + x3*2./3.
        f1, f2 = f(x1), f(x2)

        print('-----------------')
        print(f'Iteration {i}')
        print(x0, x1, x2, x3)

        # compute Lagrange polynomial using least-squares fit to 4 points, which is equivalent to the cubic Lagrange polynomial
        a0, a1, a2, a3 = polyfit4(x0, x1, x2, x3, f0, f1, f2, f3)
        print('p : ', a0, a1, a2, a3)

        # Solve the first derivative of the Lagrange polynomial for a zero
        if np.abs(a3) > 1e-9:
            delta = -3*a1*a3 + a2**2

            if delta < 0:
                x_sol = x0 if f0 < f3 else x3   # just choose the interval tha contains the minimum of the linear polynomial
                y_sol = cubic_poly(x_sol, a0, a1, a2, a3)
            else:                               # solve for the two solutions of the quadratic equation of the first derivative of the Lagrange polynomial
                x_sol_1 = (-a2 + np.sqrt(delta))/(3*a3)
                x_sol_2 = (-a2 - np.sqrt(delta))/(3*a3)

                y_sol_1 = cubic_poly(x_sol_1, a0, a1, a2, a3)
                y_sol_2 = cubic_poly(x_sol_2, a0, a1, a2, a3)

                x_sol = x_sol_1 if y_sol_1 < y_sol_2 else x_sol_2
                y_sol = y_sol_1 if y_sol_1 < y_sol_2 else y_sol_2
            
        elif np.abs(a2) > 1e-9: # if a3 is zero, then the Lagrange polynomial is a quadratic polynomial
            x_sol = -a1/(2*a2)
            y_sol = cubic_poly(x_sol, a0, a1, a2, a3)
        else:   # if a3 and a2 are zero, then the Lagrange polynomial is a linear polynomial
            x_sol = x0 if f0 < f3 else x3   # just choose the interval tha contains the minimum of the linear polynomial
            y_sol = cubic_poly(x_sol, a0, a1, a2, a3)
        
        print(f'{delta}, f({x_sol_1}) = {y_sol_1} f({x_sol_2}) = {y_sol_2}\t{x_sol}')

        if np.abs(x_sol - x_prev) < tol:
            break

        # Determine which interval contains the minimum of the cubic polynomial
        if x_sol < x1:
            x3, f3 = x1, f1
        elif x_sol < x2:
            x0, f0 = x1, f1
            x3, f3 = x2, f2
        else:
            x0, f0 = x2, f2
        
        x_prev = x_sol
    
    # return best estimate of minimum
    if y_sol < f0 and y_sol < f3:
        return x_sol
    elif f0 < f3:
        return x0
    else:
        return x3

def f(x):
    return x**2 + np.sin(5*x)
    # return (x-1)**2
    # return -x
    # return np.exp(x)
    # return -np.exp(-x**2)
    # return np.exp(-x**2)
    # return np.ones_like(x)

if __name__ == '__main__':
    def test_cubic_lagrange_polynomial():
        x = np.linspace(0, 16, 100)
        x1, x2, x3, x4 = 1, 3, 7, 15
        y1, y2, y3, y4 = 1, -2, 3.5, 5

        a0, a1, a2, a3 = polyfit4(x1, x2, x3, x4, y1, y2, y3, y4)
        print(a0, a1, a2, a3)
        y = cubic_poly(x, a0, a1, a2, a3)

        # plt.plot(x, cubic_lagrange(x, x1, x2, x3, x4, y1, y2, y3, y4), 'b', label='cubic Lagrange ChatGPT one shot')
        plt.plot(x, y, 'g', label='cubic Lagrange through coeffs polyfit style baby')
        plt.plot([x1, x2, x3, x4], [y1, y2, y3, y4], 'ro')
        plt.grid()
        plt.show()
    
    def test_cubic_lagrange_minimize():
        x = np.linspace(-2, 2, 100)
        x_min = cubic_lagrange_minimize(f, -1.2, 1.5, tol=1e-6)
        print('Solution : ', x_min, f(x_min))
        plt.plot(x, f(x), 'b')
        plt.plot(x_min, f(x_min), 'ro')
        plt.grid()
        plt.show()
    
    test_cubic_lagrange_polynomial()
    test_cubic_lagrange_minimize()