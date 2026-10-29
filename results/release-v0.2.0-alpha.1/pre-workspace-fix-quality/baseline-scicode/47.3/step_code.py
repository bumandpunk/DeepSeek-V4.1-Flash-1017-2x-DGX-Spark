def U_system(positions, sigma, epsilon):
    '''Total energy of entire system
    Inputs:
    positions: all atom positions, 2D array of floats with shape (N,3), where N is the number of atoms, 3 is for x,y,z coordinate
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    Outputs:
    U: Aggregated energy of entire system, float
    '''
    U = 0.0
    N = len(positions)

    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(np.asarray(positions[i]) - positions[j])
            U += U_ij(r, sigma, epsilon)

    return U