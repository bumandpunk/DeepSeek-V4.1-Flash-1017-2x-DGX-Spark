def orthogonalize(A):
    '''Perform Gram-Schmidt orthogonalization on the input vectors to produce orthogonal and normalized vectors.
    Input:
    A (N*N numpy array): N linearly independent vectors in the N-dimension space.
    Output:
    B (N*N numpy array): The collection of the orthonomal vectors.
    '''
    n = A.shape[0]
    B = np.zeros(A.shape, dtype=float)

    for i in range(n):
        v = A[:, i].astype(float).copy()
        for j in range(i):
            q = B[:, j]
            coeff = inner_product(A[:, i], q) / inner_product(q, q)
            v = v - coeff * q
        B[:, i] = normalize(v)

    return B