def normalize(v):
    '''Normalize the input vector.
    Input:
    v (N*1 numpy array): The input vector.
    Output:
    n (N*1 numpy array): The normalized vector.
    '''
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    n = v / norm
    return n