def tensor_d(i, j, k):
    '''Returns the tensor d_ijk of the SU(3) algebra.
    Input
    i: the first index; int
    j: the second index; int
    k: the third index; int
    Output
    result: the value of d_ijk for given indices i, j, and k; float
    '''

    def gell_mann(n):
        if n == 1:
            return np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 0]], dtype=complex)
        elif n == 2:
            return np.array([[0, -1j, 0],
                             [1j, 0, 0],
                             [0, 0, 0]], dtype=complex)
        elif n == 3:
            return np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 0]], dtype=complex)
        elif n == 4:
            return np.array([[0, 0, 1],
                             [0, 0, 0],
                             [1, 0, 0]], dtype=complex)
        elif n == 5:
            return np.array([[0, 0, -1j],
                             [0, 0, 0],
                             [1j, 0, 0]], dtype=complex)
        elif n == 6:
            return np.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0]], dtype=complex)
        elif n == 7:
            return np.array([[0, 0, 0],
                             [0, 0, -1j],
                             [0, 1j, 0]], dtype=complex)
        elif n == 8:
            sqrt3 = 1.0 / np.sqrt(3.0)
            return np.array([[sqrt3, 0, 0],
                             [0, sqrt3, 0],
                             [0, 0, -2.0 * sqrt3]], dtype=complex)
        else:
            return np.zeros((3, 3), dtype=complex)

    lambda_i = gell_mann(i)
    lambda_j = gell_mann(j)
    lambda_k = gell_mann(k)

    anticommutator = lambda_i @ lambda_j + lambda_j @ lambda_i
    result = np.real(np.trace(anticommutator @ lambda_k)) / 4.0

    return float(result)