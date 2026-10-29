def f_ij(sigma, epsilon, r):
    '''This function computes the force between two particles interacting through Lennard Jones Potantial
    Inputs:
    sigma: the distance at which potential reaches zero, float
    epsilon: potential well depth, float
    r: 3D displacement between two particles, 1d array of float with shape (3,)
    Outputs:
    f: force, 1d array of float with shape (3,)
    '''
    r_norm = np.linalg.norm(r)
    sr6 = (sigma / r_norm) ** 6
    sr12 = sr6 ** 2
    factor = 24.0 * epsilon * (2.0 * sr12 - sr6) / (r_norm ** 2)
    f = factor * r
    return f