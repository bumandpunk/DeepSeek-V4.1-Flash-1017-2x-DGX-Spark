import numpy as np
from scipy.spatial.distance import pdist

def U_ij(r, sigma, epsilon):
    '''Lennard Jones Potential between pair of atoms with distance r
    Inputs:
    r: distance, float
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float
    Outputs:
    U: Potential Energy, float
    '''

    ratio = sigma / r
    U = 4.0 * epsilon * (ratio**12 - ratio**6)
    return U
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('47.1',3,"/data/test_data.h5")

target=targets[0]
r = 0.5
sigma = 1
epsilon = 3
assert np.allclose(U_ij(r,sigma,epsilon), target)

target=targets[1]
r = 1.1
sigma = 2
epsilon = 1
assert np.allclose(U_ij(r,sigma,epsilon), target)

target=targets[2]
r = 5.0
sigma = 5
epsilon = 3
assert np.allclose(U_ij(r,sigma,epsilon), target)
