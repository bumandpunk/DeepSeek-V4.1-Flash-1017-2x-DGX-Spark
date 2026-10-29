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
def select_big_weights(gpoints_all, cell_volume, alpha, tol=1e-10):
    '''Filter g-points based on weight in reciprocal space.
    Parameters:
        gpoints_all (np.ndarray): An array of shape (nk, 3) representing all g-points.
        cell_volume (float): The volume of the unit cell.
        alpha (float): The alpha value used for the Ewald summation.
        tol (float, optional): The tolerance for filtering weights. Default is 1e-10.
    Returns:
        tuple: 
            gpoints (np.array): An array of shape (nk, 3) containing g-points with significant weights.
            gweights (np.array): An array of shape (nk,) containing the weights of the selected g-points.       
    '''
    gpoints_all = np.asarray(gpoints_all)
    k2 = np.sum(gpoints_all**2, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        gweights_all = (4.0 * np.pi / cell_volume) * np.exp(-k2 / (4.0 * alpha**2)) / k2

    mask = (k2 > 0.0) & (gweights_all > tol)

    gpoints = gpoints_all[mask]
    gweights = gweights_all[mask]

    return gpoints, gweights
def sum_recip(atom_charges, atom_coords, configs, gweights, gpoints):
    '''Calculate the reciprocal lattice sum for the Ewald summation.
    Parameters:
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        atom_coords (np.ndarray): An array of shape (natoms, 3) representing the coordinates of the atoms.
        configs (np.ndarray): An array of shape (nelectrons, 3) representing the configurations of the electrons.
        gweights (np.ndarray): An array of shape (nk,) representing the weights of the g-points.
        gpoints (np.ndarray): An array of shape (nk, 3) representing the g-points.
    Returns:
        float: The reciprocal lattice sum.
    '''
    charges = np.concatenate([atom_charges, -np.ones(len(configs))])
    coords = np.concatenate([atom_coords, configs], axis=0)

    if len(gpoints) == 0:
        return 0.0

    phases = coords @ gpoints.T
    exp_ikr = np.exp(1j * phases)
    exp_minus_ikr = np.exp(-1j * phases)

    sum_plus = np.sum(charges[:, None] * exp_ikr, axis=0)
    sum_minus = np.sum(charges[:, None] * exp_minus_ikr, axis=0)

    terms = gweights * np.abs(sum_plus * sum_minus)
    return float(np.sum(terms))
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('10.8',4,"/data/test_data.h5")

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
EX = EX1
gpoints, gweights = select_big_weights(
    generate_gpoints(np.linalg.inv(EX['latvec']).T, gmax=1), 
    np.linalg.det(EX['latvec']), 
    get_alpha(np.linalg.inv(EX['latvec']).T)
    )
assert np.allclose(sum_recip(EX['atom_charges'], EX['atom_coords'], EX['configs'], gweights, gpoints), target)

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
EX = EX2
gpoints, gweights = select_big_weights(
    generate_gpoints(np.linalg.inv(EX['latvec']).T, gmax=1), 
    np.linalg.det(EX['latvec']), 
    get_alpha(np.linalg.inv(EX['latvec']).T)
    )
assert np.allclose(sum_recip(EX['atom_charges'], EX['atom_coords'], EX['configs'], gweights, gpoints), target)

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
EX = EX3
gpoints, gweights = select_big_weights(
    generate_gpoints(np.linalg.inv(EX['latvec']).T, gmax=1), 
    np.linalg.det(EX['latvec']), 
    get_alpha(np.linalg.inv(EX['latvec']).T)
    )
assert np.allclose(sum_recip(EX['atom_charges'], EX['atom_coords'], EX['configs'], gweights, gpoints), target)

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
    ]) * L / 2,
    'configs': np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 3.0],
        [1.0, 3.0, 1.0],
        [1.0, 3.0, 3.0],
        [3.0, 1.0, 1.0],
        [3.0, 1.0, 3.0],
        [3.0, 3.0, 1.0],
        [3.0, 3.0, 3.0]
    ]) * L / 4
}
EX = EX4
gpoints, gweights = select_big_weights(
    generate_gpoints(np.linalg.inv(EX['latvec']).T, gmax=1), 
    np.linalg.det(EX['latvec']), 
    get_alpha(np.linalg.inv(EX['latvec']).T)
    )
assert np.allclose(sum_recip(EX['atom_charges'], EX['atom_coords'], EX['configs'], gweights, gpoints), target)
