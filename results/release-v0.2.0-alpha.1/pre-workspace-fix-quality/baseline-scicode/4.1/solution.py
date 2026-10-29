import numpy as np

def ichol(A):
    '''Inputs:
    A : Matrix, 2d array M * M
    Outputs:
    A : Matrix, 2d array M * M
    '''
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        diag_sum = 0.0
        for k in range(i):
            diag_sum += L[i, k] * L[i, k]

        L[i, i] = np.sqrt(A[i, i] - diag_sum)

        for j in range(i + 1, n):
            if A[j, i] != 0.0:
                off_sum = 0.0
                for k in range(i):
                    off_sum += L[i, k] * L[j, k]
                L[j, i] = (A[j, i] - off_sum) / L[i, i]

    return L