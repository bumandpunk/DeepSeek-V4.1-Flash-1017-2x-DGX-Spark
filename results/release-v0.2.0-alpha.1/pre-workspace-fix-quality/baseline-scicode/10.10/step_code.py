def sum_charge(atom_charges, nelec, cell_volume, alpha):
    '''Calculate the summation of the charge terms for the Ewald summation.
    Parameters:
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        nelec (int): The number of electrons.
        cell_volume (float): The volume of the unit cell.
        alpha (float): The alpha value used for the Ewald summation.
    Returns:
        float: The sum of the charge terms.
    '''
    atom_charges = np.asarray(atom_charges)
    total_charge = np.sum(atom_charges) - nelec
    val = -(np.pi / (2.0 * cell_volume * alpha**2)) * np.abs(total_charge)**2
    return float(val)