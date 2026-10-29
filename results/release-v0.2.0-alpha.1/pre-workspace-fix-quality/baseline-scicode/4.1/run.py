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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('4.1',3,"/data/test_data.h5")

target=targets[0]
n = 7
h = 1.0/n
diagonal = [2/h for i in range(n)]
diagonal_up = [-1/h for i in range(n-1)]
diagonal_down = [-1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
assert np.allclose(ichol(A), target)

target=targets[1]
n = 7
h = 1.0/n
diagonal = [1/h for i in range(n)]
diagonal_up = [-0.1/h for i in range(n-1)]
diagonal_down = [-0.1/h for i in range(n-1)]
A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
assert np.allclose(ichol(A), target)

target=targets[2]
n = 7
h = 1.0/n
diagonal = [2/h for i in range(n)]
diagonal_up = [-0.3/h for i in range(n-2)]
diagonal_down = [-0.3/h for i in range(n-2)]
A = np.diag(diagonal) + np.diag(diagonal_up, 2) + np.diag(diagonal_down, -2)
assert np.allclose(ichol(A), target)
