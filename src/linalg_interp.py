import numpy as np

def gauss_iter_solve(A, b, x0 = None, tol = 1e-8, alg = 'seidel'):

    """
    This function will use Gauss-Sediel iteration to solve a linear systems of equations.
    You can input an optional initial guess, error tolerance (stopping criterion), and prefrence of which algorythm to use.

    Parameters
    ----------
    A : arraylike
        Coeffecent matrix. Must be a square nxn matrix

    b : arraylike
        Right hand side vector. Must be a 1xn vector

    x0 : arraylike
        Optional initial guess vector. Defaults to guess of all 0s. Format should be same as b

    tol : float
        Optional relative error tolerance. Defaults to 1e-8

    alg : string
        Algorythm flag with accepted inputs of 'seidel' or 'jacobi'. Defaults to 'seidel'

    Returns
    -------

    numpy.ndarray : same shape as input vector b

    """

    A = np.array(A, dtype = float) #casting inputs to floats 
    b = np.array(b, dtype = float)
    
    if not x0:
        x = (b/np.trace(A)) #populating x vector if empty
    else:
        x = np.array(x0, dtype = float) #casting inputted x values to floats

    #initial error handling
    dim = len(A.shape)
    if dim != 2: #make sure the A matrix is 2d
        raise ValueError(f"A has {dim} dimentions, it should be 2")
    if A.shape[1] != len(A): #make sure A is a square matrix
        raise ValueError(f"A has {len(A)} rows and {A.shape[1]} cols, it should be square")
    dimb = len(b.shape) 
    if dimb != 1: #make sure the rhs vector is a 1d array
        raise ValueError(f"b has {dimb} dimentions, should be 1D")
    if len(b) != len(A): #make sure number of equations matches number of right hand sides
        raise ValueError(f"A has {len(A)} rows, b has {len(b)} values, dimentions are incompatable")
    if alg.lower().strip() not in ['seidel', 'jacobi']:
        raise ValueError(f"Chosen algorythm is not one of 'seidel' or 'jacobi', input is {alg}.")

    error = 2 * tol #initalize error
    n = 0 #initialize counter
    max_i = 100 #max iterations

    A_diag = np.diag(1.0/np.diag(A)) #normalize coeff matrix
    b_star = A_diag @ b
    A_star = A_diag @ A
    A_s = A_star - np.eye(len(A)) 

    #gauss seidel iteration
    if alg.lower().strip() == "seidel":

        while error > tol and n < max_i: #run gauss seidel alg until convergence or max iterations

            xcop = x.copy() #copy x
            n += 1 #incriment n
            x = b_star - A_s @ x #update x guess
            dx = x - xcop #find diffrence between new and old guess
            error = np.linalg.norm(dx) / np.linalg.norm(x) #update relative error

    if alg.lower().strip() == "jacobi":

        while error > tol and n < max_i: #do jacobi iteration until it converges or reaches max iterations

            xcop = x.copy() #copy x
            n += 1 #incriment n

            for i,a_row in enumerate(A_s): #update each x in the vector
                x[i] = b_star[i] - np.dot(a_row,x)

            dx = x - xcop #find difference between old and new guesses
            error = np.linalg.norm(dx) / np.linalg.norm(x) #update relative error
            
    if n >= max_i: #notify if max iterations has veen reached
        raise RuntimeWarning(f"System has not converged after {max_i} iterations. Returning most recent solution")

    return x

def spline_function(xd, yd, order = 3):

    """does spline interpolation in either first, second, or third order

    Parameters
    ----------
    xd : numpy.ndarray, shape=(n,1)
        x values in ascending order
    yd : numpy.ndarray, shape=(n,1)
        y values
    order : int, 1 or 2 or 3
        desired order of interpolating function

    Returns
    -------
    spline order : function
        interpolating function
    """

    k_sort = np.argsort(xd) #sorted array of coefficents
    xd = np.array([xd[k] for k in k_sort]) #sort x in ascending order
    yd = np.array([yd[k] for k in k_sort]) #sort y based on x

    if (xd.shape != yd.shape): #checking x and y are the same shape
        raise ValueError(f"xd length = {len(xd)} yd length = {len(yd)} both must be the same length")

    if order not in (1,2,3): #checking that the desired order is within spec
        raise ValueError(f"{order} order not supported.")
    
    N = len(xd) #number of datapoints
    a = yd[:-1] #trimmed y values
    dx = np.diff(xd) #vector of x diffrences    
    dy = np.diff(yd) #vector of y diffrences
    f1 = dy/dx #first order diffrence 

    if order == 1:

        b = f1 #set right hand side vector
        def s1(x): 

            k = (0 if x <=  xd[0] #assigning location of the spline based on x value
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])

            return a[k] + b[k] * (x - xd[k]) #interpolated value at x
        
        return s1
    
    if order == 2:
        
        A0 = np.hstack([np.diag(dx[:-1]),np.zeros((N-2,1))]) #put 0 vector to left side of lower triangle
        A1 = np.hstack([np.zeros((N-2,1)),np.diag(dx[1:])]) #put 0 vector to right side of lower triangle
        A = np.vstack([np.zeros((1,N-1)),(A0+A1)]) #put row of 0 above A0 and A1 matrix
        B = np.zeros((N-1,)) #set B
        B[1:] = np.diff(f1) #find second order diffrences
        A[0,:2] = [1,-1] #add constants to first row

        c = np.linalg.solve(A,B) #solve for 2nd order coefficent
        b = f1 - c * dx #find right hand side vector

        def s2(x): #find spline location for x value location

            k = (0 if x <=  xd[0]
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])
            
            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 #return interpolated x value

        return s2
    
    if order == 3:
        
        A = np.zeros((N,N)) # make empty N by N matrix
        A[1:-1,:-2] += np.diag(dx[:-1]) #adding lower triangular portion
        A[1:-1,1:-1] += np.diag(2 * (dx[:-1] + dx[1:])) #adding main diagona
        A[1:-1,2:] += np.diag(dx[1:]) #add upper triangular portion
        A[0,:3] = [-dx[1],dx[0]+dx[1],-dx[0]] #adding first row
        A[-1,-3:] = [-dx[-1],dx[-1]+dx[-2],-dx[-2]] #add last row

        B = np.zeros((N,))
        B[1:-1] = 3 * np.diff(f1) 

        c = gauss_iter_solve(A,B) #using gauss seidel to solve for 2nd order coefficent
        d = np.diff(c) / (3 * dx) #third order coefficent
        b = f1 - c[:-1] * dx - d * dx ** 2 #updating right hand side vector
        
        def s3(x): #find spline location from x value

            k = (0 if x <=  xd[0] 
                 else len(a) - 1 if x >= xd[-1]
                else np.nonzero(xd<x)[0][-1])

            return a[k] + b[k] * (x - xd[k]) + c[k] * (x - xd[k]) ** 2 + d[k] * (x - xd[k]) ** 3 #return interpolated x

        return s3