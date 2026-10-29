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