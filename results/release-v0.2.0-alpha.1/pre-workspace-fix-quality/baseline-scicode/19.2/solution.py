import numpy as np
from scipy.linalg import sqrtm
def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of arrays, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices
    '''
    if len(args) == 0:
        return np.array(1.0)

    M = args[0]
    for matrix in args[1:]:
        M = np.kron(M, matrix)

    return M
def n_tangle(psi):
    '''Returns the n_tangle of pure state psi
    Input:
    psi: 1d array of floats, the vector representation of the state
    Output:
    tangle: float, the n-tangle of psi
    '''
    psi = np.asarray(psi, dtype=complex)
    n = int(np.round(np.log2(psi.size)))

    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    operator = np.array([[1.0 + 0j]], dtype=complex)
    for _ in range(n):
        operator = np.kron(operator, sigma_y)

    amplitude = np.vdot(psi, operator @ np.conjugate(psi))
    tangle = np.abs(amplitude) ** 2

    return float(tangle)