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