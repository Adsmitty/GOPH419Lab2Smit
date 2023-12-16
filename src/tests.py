import numpy as np
import matplotlib.pyplot as plt

from linalg_interp import gauss_iter_solve
from linalg_interp import spline_function

def main():

    A = np.array([[9, 1, 0, 0], #creating a test coefficent matrix
                [0, 12, 7, 0],
                [0, 0, 8, 7],
                [0, 0, 0, 7]])

    b = np.array([1, 2, 3, 4]) #creating a test right hand side vector

    jacobi = gauss_iter_solve(A, b, alg = 'jacobi') 
    seidel = gauss_iter_solve(A, b, alg = "seidel") 
    exp = np.linalg.solve(A, b) 

    #printing test results
    print(f"Expected = {exp}")
    print(f"Seidel = {seidel}")
    print(f"Jacobi = {jacobi}")

    #calculating spline interpolations for orders 1, 2, and 3
    x1 = np.linspace(-5, 5, 10)
    y1 = 12 + 55 * x1
    s1 = spline_function(x1, y1, order = 1)
    xp1 = np.linspace(-8, 8, 100)
    ype1 = 12 + 55 * xp1
    ypa1 = np.array([s1(x) for x in xp1])
    e1 = np.linalg.norm(ype1 - ypa1) / np.linalg.norm(ype1)
    
    x2 = np.linspace(-5, 5, 5)
    y2 = 12 + 55 * x2 + 21 * x2 ** 2
    s2 = spline_function(x2, y2, order = 2)
    xp2 = np.linspace(-8, 8, 100)
    ype2 = 12 + 55 * xp2 + 21 * xp2 ** 2
    ypa2 = np.array([s2(x) for x in xp2])
    e2 = np.linalg.norm(ype2 - ypa2) / np.linalg.norm(ype2)

    x3 = np.linspace(-5, 5, 10)
    y3 = 12 + 55 * x3 + 21 * x3 ** 2 - 4 * x3 ** 3
    s3 = spline_function(x3, y3, order = 3)
    xp3 = np.linspace(-8, 8, 100)
    ype3 = 12 + 55 * xp3 + 21 * xp3 ** 2 - 4 * xp3 ** 3
    ypa3 = np.array([s3(x) for x in xp3])
    e3 = np.linalg.norm(ype3 - ypa3) / np.linalg.norm(ype3)

    #plotting each test order on a saperate graph
    plt.figure()
    plt.plot(x1, y1, linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "blue", label = "Data")
    plt.plot(xp1, ypa1, linestyle = "dashed", color = "black", linewidth = 2, label = "1st order spline")
    plt.text(1,12,f"eps_t = {e1}")
    plt.legend()
    plt.savefig("Figures/lineartest.png")

    plt.figure()
    plt.plot(x2, y2, linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "blue", label = "Data")
    plt.plot(xp2, ypa2, linestyle = "dashed", color = "black", linewidth = 2, label = "2nd order spline")
    plt.text(1, 12, f"eps_t = {e2}")
    plt.legend()
    plt.savefig("Figures/quadratictest.png")

    plt.figure()
    plt.plot(x3, y3, linestyle = "none", marker = "x", markersize = 7, markeredgecolor = "blue", label = "Data")
    plt.plot(xp3, ypa3, linestyle = "dashed", color = "black", linewidth = 2, label = "3rd order spline")
    plt.text(1, 12, f"eps_t = {e3}")
    plt.legend()
    plt.savefig("Figures/cubictest.png")

if __name__ == "__main__":

    main()