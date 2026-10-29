import numpy as np

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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('51.1',3,"/data/test_data.h5")

target=targets[0]
sigma = 1
epsilon = 1
r = np.array([-3.22883506e-03,  2.57056485e+00,  1.40822287e-04])
assert np.allclose(f_ij(sigma,epsilon,r), target)

target=targets[1]
sigma = 2
epsilon = 3
r = np.array([3,  -4,  5])
assert np.allclose(f_ij(sigma,epsilon,r), target)

target=targets[2]
sigma = 3
epsilon = 7
r = np.array([5,  9,  7])
assert np.allclose(f_ij(sigma,epsilon,r), target)
