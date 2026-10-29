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