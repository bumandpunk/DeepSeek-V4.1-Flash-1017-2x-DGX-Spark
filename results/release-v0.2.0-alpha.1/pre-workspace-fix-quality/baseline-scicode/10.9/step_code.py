def sum_real_self(atom_charges, nelec, alpha):
    '''Calculate the real-space summation of the self terms for the Ewald summation.
    Parameters:
        atom_charges (np.ndarray): An array of shape (natoms,) representing the charges of the atoms.
        nelec (int): The number of electrons.
        alpha (float): The alpha value used for the Ewald summation.
    Returns:
        float: The real-space sum of the self terms.
    '''
    atom_charges = np.asarray(atom_charges)
    sum_q_squared = np.sum(atom_charges**2) + nelec
    val = -(alpha / np.sqrt(np.pi)) * sum_q_squared
    return float(val)