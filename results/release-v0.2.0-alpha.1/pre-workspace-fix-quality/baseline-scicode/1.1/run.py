import numpy as np

def cg(A, b, x, tol):
    '''Inputs:
    A : Matrix, 2d array size M * M
    b : Vector, 1d array size M
    x : Initial guess vector, 1d array size M
    tol : tolerance, float
    Outputs:
    x : solution vector, 1d array size M
    '''
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float)

    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = np.dot(r, r)

    n = len(b)
    if np.sqrt(rs_old) <= tol:
        return x

    for _ in range(n):
        Ap = np.dot(A, p)
        pAp = np.dot(p, Ap)

        if pAp == 0:
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) <= tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('1.1',3,"/data/test_data.h5")

target=targets[0]
n = 7
h = 1.0/n
diagonal = [2/h for i in range(n)]
diagonal_up = [-1/h for i in range(n-1)]
diagonal_down = [-1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
x0 = np.zeros(n)
tol = 10e-5
assert np.allclose(cg(A, b, x0,tol), target)

target=targets[1]
n = 7
h = 1.0/n
diagonal = [1/h for i in range(n)]
diagonal_up = [-9/h for i in range(n-1)]
diagonal_down = [-9/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
A[:, 0] = 0
A[0, :] = 0
A[0, 0] = 1/h
b = np.array([0.1,0.1,0.0,10,0.0,0.1,0.1])
x0 = np.zeros(n)
maxIter = 200
tol = 10e-7
assert np.allclose(cg(A, b, x0,tol), target)

target=targets[2]
n = 7
h = 1.0/n
diagonal = [1/h for i in range(n)]
diagonal_up = [-0.9/h for i in range(n-1)]
diagonal_down = [-0.9/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
b = np.array([0.1,10.1,0.0,0.5,0.2,0.3,0.5])
x0 = np.zeros(n)
maxIter = 500
tol = 10e-7
assert np.allclose(cg(A, b, x0,tol), target)
