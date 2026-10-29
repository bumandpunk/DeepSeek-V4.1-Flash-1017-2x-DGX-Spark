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
def distance_matrix(configs):
    '''Args:
        configs (np.array): (nparticles, 3)
    Returns:
        distances (np.array): distance vector for each particle pair. Shape (npairs, 3), where npairs = (nparticles choose 2)
        pair_idxs (list of tuples): list of pair indices
    '''
    nparticles = len(configs)
    idx_i, idx_j = np.triu_indices(nparticles, k=1)
    pair_idxs = list(zip(idx_i.tolist(), idx_j.tolist()))
    distances = configs[idx_i] - configs[idx_j]
    return distances, pair_idxs

def sum_real_cross(atom_charges, atom_coords, configs, lattice_coords, alpha):
    '''Calculate the sum of real-space cross terms for the Ewald summation.
    Parameters:
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        atom_coords (np.ndarray): An array of shape (natoms, 3) representing the coordinates of the atoms.
        configs (np.ndarray): An array of shape (nelectrons, 3) representing the configurations of the electrons.
        lattice_coords (np.ndarray): An array of shape (ncells, 3) representing the lattice coordinates.
        alpha (float): The alpha value used for the Ewald summation.
    Returns:
        float: The sum of ion-ion, electron-ion, and electron-electron cross terms.
    '''
    # Combine ions and electrons into a single particle set.
    charges = np.concatenate([atom_charges, -np.ones(len(configs))])
    coords = np.concatenate([atom_coords, configs], axis=0)

    distances, pair_idxs = distance_matrix(coords)

    if len(pair_idxs) == 0:
        return 0.0

    # Reshape for real_cij: (batch, npairs, 1, 3) and (batch, 1, ncells, 3).
    d = distances[np.newaxis, :, np.newaxis, :]
    lc = lattice_coords[np.newaxis, np.newaxis, :, :]

    cij = real_cij(d, lc, alpha)
    cij = np.asarray(cij).reshape(-1)

    pair_idxs = np.array(pair_idxs)
    qi = charges[pair_idxs[:, 0]]
    qj = charges[pair_idxs[:, 1]]

    return float(np.sum(qi * qj * cij))
def generate_gpoints(recvec, gmax):
    '''Generate a grid of g-points for reciprocal space based on the provided lattice vectors.
    Parameters:
        recvec (np.ndarray): A 3x3 array representing the reciprocal lattice vectors.
        gmax (int): The maximum integer number of lattice points to include in one positive direction.
    Returns:
        np.ndarray: An array of shape (nk, 3) representing the grid of g-points.
    '''
    x_vals = np.arange(0, gmax + 1)
    yz_vals = np.arange(-gmax, gmax + 1)

    grid = np.meshgrid(x_vals, yz_vals, yz_vals, indexing='ij')
    gpoints = np.stack(grid, axis=-1).reshape(-1, 3)

    x = gpoints[:, 0]
    y = gpoints[:, 1]
    z = gpoints[:, 2]

    valid = (
        (x > 0)
        | ((x == 0) & (y > 0))
        | ((x == 0) & (y == 0) & (z > 0))
    )

    gpoints_all = gpoints[valid] @ recvec
    return gpoints_all