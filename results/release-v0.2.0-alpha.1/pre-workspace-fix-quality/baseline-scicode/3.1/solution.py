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