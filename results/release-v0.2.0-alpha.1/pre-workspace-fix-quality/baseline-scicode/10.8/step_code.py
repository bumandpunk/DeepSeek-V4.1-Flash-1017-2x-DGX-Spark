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