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