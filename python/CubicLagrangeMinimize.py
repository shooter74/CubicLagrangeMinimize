# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cubic_poly(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

def polyfit4(x1, x2, x3, x4, y1, y2, y3, y4):
    ''' Function to compute the coefficients of the cubic Lagrange polynomial that interpolates the four points (x1, y1), (x2, y2), (x3, y3), (x4, y4).
        The coefficients are computed using a linear least-squares fit to the four points.
        The condition number of the matrix (A^T * A) is critical to the numerical stability of the solution.
        It happens that the minimum condition number is achieved when the four points are equally spaced and span the interval [-pi/3, pi/3].
    '''
    A = np.array([[1, x1, x1**2, x1**3], [1, x2, x2**2, x2**3], [1, x3, x3**2, x3**3], [1, x4, x4**2, x4**3]])
    y = np.array([y1, y2, y3, y4])
    ATA = A.transpose()@A
    # print(ATA)# DEBUG
    # print(np.linalg.matrix_rank(ATA))# DEBUG
    # print(np.linalg.det(ATA))# DEBUG
    a = np.linalg.solve(A.transpose()@A, A.transpose()@y)
    return a

def map_interval(x, a1, b1, a2, b2):
    ''' Maps a value or array "x" from the interval [a1, b1] to the interval [a2, b2]. '''
    return (x - a1) * (b2 - a2) / (b1 - a1) + a2

def cubic_lagrange_minimize(f, a, b, tol=1e-6, callback=None):
    ''' Function to find minimum of f over interval [a, b] using cubic Lagrange polynomial interpolation.
        If the function is monotonic, then the minimum is one of the bounds of the interval, and the minimum is found in two iterations.
        The best estimate of the minimum is returned.
        The number of function evaluations is 2 + 2*Niter.
        callback is a function that is called at each iteration with the current estimate of the minimum and other internal parameters of the algorithm.
        The callback function should have the following signature: callback(x_sol, y_sol, i, quadratic_solution, x, a), where quadratic_solution = (delta, x_sol_1, x_sol_2, y_sol_1, y_sol_2) is the solution of the quadratic equation of the first derivative of the Lagrange polynomial, x = (x0, x1, x2, x3) are the endpoints of the interval, and a = (a0, a1, a2, a3) are the coefficients of the Lagrange polynomial.
    '''
    # initialize interval endpoints and function values
    x0, x1, x2, x3 = a, a*2./3. + b*1./3., a*1./3. + b*2./3., b
    f0, f3 = f(x0), f(x3)
    x_prev = (x0 + x3)/2.
    small_coefficient = 1e-9
    Niter = int(np.ceil(np.log(np.abs(b-a)/tol)/np.log(3)))# reduction factor of interval size at each iteration is 3 because we sample two points in the interval each iteration

    for i in range(Niter):
        # Compute function values at two points in the interval
        x1, x2 = x0*2./3. + x3*1./3., x0*1./3. + x3*2./3.
        f1, f2 = f(x1), f(x2)

        # Remap the x values to the interval [-pi/3, pi/3] to minimize the condition number of the matrix (A^T * A) in the least-squares fit
        x0m, x1m, x2m, x3m = map_interval(x0, x0, x3, -np.pi/3, np.pi/3), map_interval(x1, x0, x3, -np.pi/3, np.pi/3), map_interval(x2, x0, x3, -np.pi/3, np.pi/3), map_interval(x3, x0, x3, -np.pi/3, np.pi/3)

        # compute Lagrange polynomial using least-squares fit to 4 points, which is equivalent to the cubic Lagrange polynomial
        a0, a1, a2, a3 = polyfit4(x0m, x1m, x2m, x3m, f0, f1, f2, f3)
        x_sol_1, x_sol_2, y_sol_1, y_sol_2, delta = None, None, None, None, None

        # Solve the first derivative of the Lagrange polynomial for a zero
        if np.abs(a3) > small_coefficient:
            delta = -3*a1*a3 + a2**2

            if delta < 0:
                x_sol = x0m if f0 < f3 else x3m   # just choose the interval tha contains the minimum of the linear polynomial
                y_sol = cubic_poly(x_sol, a0, a1, a2, a3)
            else:                               # solve for the two solutions of the quadratic equation of the first derivative of the Lagrange polynomial
                x_sol_1 = (-a2 + np.sqrt(delta))/(3*a3)
                x_sol_2 = (-a2 - np.sqrt(delta))/(3*a3)

                y_sol_1 = cubic_poly(x_sol_1, a0, a1, a2, a3)
                y_sol_2 = cubic_poly(x_sol_2, a0, a1, a2, a3)

                x_sol = x_sol_1 if y_sol_1 < y_sol_2 else x_sol_2
                y_sol = y_sol_1 if y_sol_1 < y_sol_2 else y_sol_2
            
        elif np.abs(a2) > small_coefficient: # if a3 is zero, then the Lagrange polynomial is a quadratic polynomial
            x_sol = -a1/(2*a2)
            y_sol = cubic_poly(x_sol, a0, a1, a2, a3)
        else:   # if a3 and a2 are zero, then the Lagrange polynomial is a linear polynomial
            x_sol = x0m if f0 < f3 else x3m   # just choose the interval tha contains the minimum of the linear polynomial
            y_sol = cubic_poly(x_sol, a0, a1, a2, a3)
        
        # transform the solution back to the original interval
        x_sol = map_interval(x_sol, -np.pi/3, np.pi/3, x0, x3)
        if x_sol_1 is not None:
            x_sol_1 = map_interval(x_sol_1, -np.pi/3, np.pi/3, x0, x3)
        if x_sol_2 is not None:
            x_sol_2 = map_interval(x_sol_2, -np.pi/3, np.pi/3, x0, x3)

        if callback is not None:
            callback(x_sol, y_sol, i, (delta, x_sol_1, x_sol_2, y_sol_1, y_sol_2), (x0, x1, x2, x3), (a0, a1, a2, a3))

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

def cubic_lagrange_minimize_callback_simple(x_sol, y_sol, i, quadratic_solution, x, a):
    print(f'Iteration {i} : f({x_sol}) = {y_sol}')

def cubic_lagrange_minimize_callback_detailed(x_sol, y_sol, i, quadratic_solution, x, a):
    print('--------------------------------------------------------------------')
    print(f'Iteration {i}')
    print('X values : ', x)
    print('Cubic poly coeffs : ', a)
    print(f'Quadratic solution : delta = {quadratic_solution[0]}, x_sol_1 = {quadratic_solution[1]}, x_sol_2 = {quadratic_solution[2]}, y_sol_1 = {quadratic_solution[3]}, y_sol_2 = {quadratic_solution[4]}')
    print(f'Current solution : f({x_sol}) = {y_sol}')

if __name__ == '__main__':
    def f(x):
        # return x**2 + np.sin(5*x)
        # return (x-1)**2
        # return -x
        # return np.exp(x)
        # return np.exp(-x)
        # return -np.exp(-x**2)
        return np.exp(-x**2)
        # return np.ones_like(x)

    def test_cubic_lagrange_polynomial():
        x = np.linspace(0, 16, 100)
        x1, x2, x3, x4 = 1, 3, 7, 15
        y1, y2, y3, y4 = 1, -2, 3.5, 5

        a0, a1, a2, a3 = polyfit4(x1, x2, x3, x4, y1, y2, y3, y4)
        print(a0, a1, a2, a3)
        y = cubic_poly(x, a0, a1, a2, a3)

        # plt.plot(x, cubic_lagrange(x, x1, x2, x3, x4, y1, y2, y3, y4), 'b', label='cubic Lagrange ChatGPT one shot')
        plt.plot(x, y, 'g', label='Cubic Lagrange polynomial through linear least-squares')
        plt.plot([x1, x2, x3, x4], [y1, y2, y3, y4], 'ro')
        plt.grid()
        plt.show()
    
    def test_cubic_lagrange_minimize():
        a, b = -1.2, 1.5
        x = np.linspace(a, b, 100)
        x_min = cubic_lagrange_minimize(f, a, b, tol=1e-6, callback=cubic_lagrange_minimize_callback_detailed)
        print('Solution : ', x_min, f(x_min))
        plt.plot(x, f(x), 'b')
        plt.plot(x_min, f(x_min), 'ro')
        plt.grid()
        plt.show()
    
    # test_cubic_lagrange_polynomial()
    test_cubic_lagrange_minimize()