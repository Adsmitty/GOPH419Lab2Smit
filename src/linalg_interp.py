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

    A = np.array(A, dtype = float)
    b = np.array(b, dtype = float)
    
    if not x0:
        x0 = (b/np.trace(A))
    else:
        x0 = np.array(x0, dtype = float)

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
    
    #gauss-seidel iteration

    for i,_ in enumerate(x0):
        x0[i,:] = (b[i,:] - A[:,:i] @ x0[:i,:] - A[:,i+1:] @ x0[i+1:,:]) / A[i,i]
    

    #jacobi iteration

    Adinv = np.diag(1/np.trace(A))
    
    for i,_ in enumerate(x0):
        As = A - np.linalg.inv(Adinv)
        x0 = Adinv * (b - As * x0)
