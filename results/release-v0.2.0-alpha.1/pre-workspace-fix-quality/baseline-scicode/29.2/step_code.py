def inner_product(u, v):
    '''Calculates the inner product of two vectors.
    Input:
    u (numpy array): Vector 1.
    v (numpy array): Vector 2.
    Output:
    p (float): Inner product of the vectors.
    '''
    p = float(np.sum(u * v))
    return p