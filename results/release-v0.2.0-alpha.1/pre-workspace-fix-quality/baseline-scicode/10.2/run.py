import numpy as np
from scipy.special import erfc
def get_alpha(recvec, alpha_scaling=5):
    '''Calculate the alpha value for the Ewald summation, scaled by a specified factor.
    Parameters:
        recvec (np.ndarray): A 3x3 array representing the reciprocal lattice vectors.
        alpha_scaling (float): A scaling factor applied to the alpha value. Default is 5.
    Returns:
        float: The calculated alpha value.
    '''

    alpha = np.sqrt(np.max(np.sum(recvec**2, axis=1)))
    return alpha * alpha_scaling
def get_lattice_coords(latvec, nlatvec=1):
    '''Generate lattice coordinates based on the provided lattice vectors.
    Parameters:
        latvec (np.ndarray): A 3x3 array representing the lattice vectors.
        nlatvec (int): The number of lattice coordinates to generate in each direction.
    Returns:
        np.ndarray: An array of shape ((2 * nlatvec + 1)^3, 3) containing the lattice coordinates.
    '''
    lattice_indices = np.arange(-nlatvec, nlatvec + 1)
    grid = np.meshgrid(lattice_indices, lattice_indices, lattice_indices, indexing='ij')
    indices = np.stack(grid, axis=-1).reshape(-1, 3)
    lattice_coords = indices @ latvec
    return lattice_coords
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('10.2',4,"/data/test_data.h5")

target=targets[0]
ref1 = -1.74756
EX1 = {
    'latvec': np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
        ]),
    'atom_charges': np.array([1]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0]
        ]),
    'configs': np.array([
        [1.0, 1.0, 1.0]
    ]),
}
assert np.allclose(get_lattice_coords(EX1['latvec']), target)

target=targets[1]
ref2 = -6.99024
EX2 = {
    'latvec': np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
        ]),
    'atom_charges': np.array([1, 1, 1, 1]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
        ]),
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
        ])    
}
assert np.allclose(get_lattice_coords(EX2['latvec']), target)

target=targets[2]
ref3 = -5.03879
L = 4 / 3**0.5
EX3 = {
    'latvec': (np.ones((3, 3)) - np.eye(3)) * L / 2,
    'atom_charges': np.array([2]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0]
        ]),
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],        
        ]) * L/4
    }
assert np.allclose(get_lattice_coords(EX3['latvec']), target)

target=targets[3]
ref4 = -20.15516
EX4 = {
    'latvec': np.eye(3, 3) * L,
    'atom_charges': np.array([2, 2, 2, 2]),
    'atom_coords': np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0] 
    ]) * L/2,
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 3.0],
        [1.0, 3.0, 1.0],
        [1.0, 3.0, 3.0],
        [3.0, 1.0, 1.0],
        [3.0, 1.0, 3.0],
        [3.0, 3.0, 1.0],
        [3.0, 3.0, 3.0]        
    ]) * L/4
}
assert np.allclose(get_lattice_coords(EX4['latvec']), target)
