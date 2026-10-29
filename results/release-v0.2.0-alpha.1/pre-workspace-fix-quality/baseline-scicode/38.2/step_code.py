def reciprocal_3D(a1, a2, a3):
    '''Calculates the 3D reciprocal vectors given the input.
    Input:
    a1 (numpy array): Vector 1.
    a2 (numpy array): Vector 2.
    a3 (numpy array): Vector 3.
    Returns:
    b_i (list of numpy arrays): The collection of reciprocal vectors.
    '''
    denom = np.dot(a1, cross(a2, a3))
    b_i = [
        2 * np.pi * cross(a2, a3) / denom,
        2 * np.pi * cross(a3, a1) / denom,
        2 * np.pi * cross(a1, a2) / denom
    ]
    return b_i