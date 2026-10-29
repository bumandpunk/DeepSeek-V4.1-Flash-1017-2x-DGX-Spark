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
def U_i(r_i, i, positions, sigma, epsilon):
    '''Total energy on a single atom
    Inputs:
    r_i: atom position, 1d array of floats with x,y,z coordinate
    i: atom index, int
    positions: all atom positions, 2D array of floats with shape (N,3), where N is the number of atoms, 3 is x,y,z coordinate
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    Outputs:
    U_i: Aggregated energy on particle i, float
    '''
    total_energy = 0.0

    for j in range(len(positions)):
        if j == i:
            continue
        distance = np.linalg.norm(np.asarray(r_i) - positions[j])
        total_energy += U_ij(distance, sigma, epsilon)

    return total_energy
def U_system(positions, sigma, epsilon):
    '''Total energy of entire system
    Inputs:
    positions: all atom positions, 2D array of floats with shape (N,3), where N is the number of atoms, 3 is for x,y,z coordinate
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    Outputs:
    U: Aggregated energy of entire system, float
    '''
    U = 0.0
    N = len(positions)

    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(np.asarray(positions[i]) - positions[j])
            U += U_ij(r, sigma, epsilon)

    return U
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('47.3',3,"/data/test_data.h5")

target=targets[0]
positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
sigma = 1.0
epsilon = 1.0
assert np.allclose(U_system(positions,sigma,epsilon), target)

target=targets[1]
positions = np.array([[0.1, 1.5, 0.3], [0.8, 0.0, 0.0], [1.5, 1.5, 0.0], [2.5, 2.5, 0.0]])
sigma = 1.0
epsilon = 2.0
assert np.allclose(U_system(positions,sigma,epsilon), target)

target=targets[2]
positions = np.array([[0.0, 0.0, 0.0], [0.0, 7.0, 0.0], [10, 0.0, 0.0]])
sigma = 1.0
epsilon = 1.0
assert np.allclose(U_system(positions,sigma,epsilon), target)
