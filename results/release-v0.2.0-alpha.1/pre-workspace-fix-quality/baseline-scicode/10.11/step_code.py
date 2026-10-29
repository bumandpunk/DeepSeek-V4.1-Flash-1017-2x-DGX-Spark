def total_energy(latvec, atom_charges, atom_coords, configs, gmax):
    '''Calculate the total energy using the Ewald summation method.
    Parameters:
        latvec (np.ndarray): A 3x3 array representing the lattice vectors.
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        atom_coords (np.ndarray): An array of shape (natoms, 3) representing the coordinates of the atoms.
        configs (np.ndarray): An array of shape (nelectrons, 3) representing the configurations of the electrons.
        gmax (int): The maximum integer number of lattice points to include in one positive direction.
    Returns:
        float: The total energy from the Ewald summation.
    '''
    latvec = np.asarray(latvec)
    atom_charges = np.asarray(atom_charges)
    atom_coords = np.asarray(atom_coords)
    configs = np.asarray(configs)

    cell_volume = np.abs(np.linalg.det(latvec))
    recvec = 2.0 * np.pi * np.linalg.inv(latvec).T

    alpha = get_alpha(recvec)
    lattice_coords = get_lattice_coords(latvec, nlatvec=1)

    gpoints = generate_gpoints(recvec, gmax)
    gpoints, gweights = select_big_weights(gpoints, cell_volume, alpha)

    nelec = len(configs)

    energy = 0.0
    energy += sum_real_cross(atom_charges, atom_coords, configs, lattice_coords, alpha)
    energy += sum_real_self(atom_charges, nelec, alpha)
    energy += sum_recip(atom_charges, atom_coords, configs, gweights, gpoints)
    energy += sum_charge(atom_charges, nelec, cell_volume, alpha)

    return float(energy)