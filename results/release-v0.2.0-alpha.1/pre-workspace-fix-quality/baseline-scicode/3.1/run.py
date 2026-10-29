import numpy as np

def GS(A, b, eps, x_true, x0):
    '''Solve a given linear system Ax=b Gauss-Seidel iteration
    Input
    A:      N by N matrix, 2D array
    b:      N by 1 right hand side vector, 1D array
    eps:    Float number indicating error tolerance
    x_true: N by 1 true solution vector, 1D array
    x0:     N by 1 zero vector, 1D array
    Output
    residual: Float number shows L2 norm of residual (||Ax - b||_2)
    errors:   Float number shows L2 norm of error vector (||x-x_true||_2) 
    '''

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    x_true = np.asarray(x_true, dtype=float)

    n = len(b)

    while True:
        x_old = x.copy()

        for i in range(n):
            sigma = 0.0

            for j in range(i):
                sigma += A[i, j] * x[j]

            for j in range(i + 1, n):
                sigma += A[i, j] * x_old[j]

            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old) < eps:
            break

    residual = np.linalg.norm(np.dot(A, x) - b)
    error = np.linalg.norm(x - x_true)

    return residual, error
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('3.1',3,"/data/test_data.h5")

target=targets[0]
n = 7
h = 1/(n-1)
diagonal = [2/h for i in range(n)]
diagonal_up = [-1/h for i in range(n-1)]
diagonal_down = [-1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
A[:, 0] = 0
A[0, :] = 0
A[0, 0] = 1/h
A[:, -1] = 0
A[-1, :] = 0
A[7-1, 7-1] = 1/h
b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
x_true = np.linalg.solve(A, b)
eps = 10e-5
x0 = np.zeros(n)
assert np.allclose(GS(A, b, eps, x_true, x0), target)

target=targets[1]
n = 7
h = 1/(n-1)
diagonal = [2/h for i in range(n)]
diagonal_up = [-0.5/h for i in range(n-2)]
diagonal_down = [-0.5/h for i in range(n-2)]
A = np.diag(diagonal) + np.diag(diagonal_up, 2) + np.diag(diagonal_down, -2)
b = np.array([0.5,0.1,0.5,0.1,0.5,0.1,0.5])
x_true = np.linalg.solve(A, b)
eps = 10e-5
x0 = np.zeros(n)
assert np.allclose(GS(A, b, eps, x_true, x0), target)

target=targets[2]
n = 7
h = 1/(n-1)
diagonal = [2/h for i in range(n)]
diagonal_2up = [-0.5/h for i in range(n-2)]
diagonal_2down = [-0.5/h for i in range(n-2)]
diagonal_1up = [-0.3/h for i in range(n-1)]
diagonal_1down = [-0.5/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_2up, 2) + np.diag(diagonal_2down, -2) + np.diag(diagonal_1up, 1) + np.diag(diagonal_1down, -1)
b = np.array([0.5,0.1,0.5,0.1,-0.1,-0.5,-0.5])
x_true = np.linalg.solve(A, b)
eps = 10e-5
x0 = np.zeros(n)
assert np.allclose(GS(A, b, eps, x_true, x0), target)
