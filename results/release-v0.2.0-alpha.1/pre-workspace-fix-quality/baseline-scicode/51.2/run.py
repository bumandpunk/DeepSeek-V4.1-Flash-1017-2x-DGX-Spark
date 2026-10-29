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
def aggregate_forces(sigma, epsilon, positions):
    '''This function aggregates the net forces on each atom for a system of atoms interacting through Lennard Jones Potential
    Inputs:
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    positions: 3D positions of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    Outputs:
    forces: net forces on each atom, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    '''
    positions = np.asarray(positions)
    forces = np.zeros_like(positions, dtype=float)
    n_atoms = positions.shape[0]

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = positions[i] - positions[j]
            f = f_ij(sigma, epsilon, r)
            forces[i] += f
            forces[j] -= f

    return forces
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('51.2',3,"/data/test_data.h5")

target=targets[0]
sigma = 1
epsilon = 1
positions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])
assert np.allclose(aggregate_forces(sigma,epsilon,positions), target)

target=targets[1]
sigma = 2
epsilon = 3
positions = np.array([[2.47984377, 3.50604547, 0.1682584 ],[0.07566928, 5.56353376, 1.95393567], [0.57271561, 0.19380897, 0.3937953 ]])
assert np.allclose(aggregate_forces(sigma,epsilon,positions), target)

target=targets[2]
sigma = 1
epsilon = 5
positions = np.array([[.62726631, 5.3077771 , 7.29719649],
       [7.25031287, 7.58926428, 2.71262908],
       [8.7866416 , 3.73724676, 9.22676027],
       [0.89096788, 5.3872004 , 7.95350911],
       [6.068183  , 3.55807037, 2.7965242 ]])
assert np.allclose(aggregate_forces(sigma,epsilon,positions), target)
