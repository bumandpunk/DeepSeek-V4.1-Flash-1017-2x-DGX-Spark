def MC(sigma, epsilon, T, init_positions, MC_steps, dispSize):
    '''Markov Chain Monte Carlo simulation to generate samples energy of system of atoms interacting through Lennard Jones potential
    at temperature T, using Metropolis-Hasting Algorithm with Gaussian trial move
    Inputs:
    sigma: the distance at which Lennard Jones potential reaches zero, float,
    epsilon: potential well depth of Lennard Jones potential, float 
    T: Temperature in reduced unit, float
    init_position: initial position of all atoms, 2D numpy array of float with shape (N, 3) where N is number of atoms, 3 is for x,y,z coordinate
    MC_steps: Number of MC steps to perform, int
    dispSize: Size of displacement in Gaussian trial move, float
    Outputs:
    E_trace: Samples of energy obtained by MCMC sampling start with initial energy, list of floats with length MC_steps+1 
    '''
    positions = np.array(init_positions, dtype=float, copy=True)
    N = len(positions)
    E_trace = [float(U_system(positions, sigma, epsilon))]

    for _ in range(MC_steps):
        i = np.random.randint(0, N)

        old_energy_i = U_i(positions[i], i, positions, sigma, epsilon)

        displacement = np.random.normal(0.0, dispSize, size=3)
        new_position = positions[i] + displacement
        new_energy_i = U_i(new_position, i, positions, sigma, epsilon)

        delta_E = new_energy_i - old_energy_i

        if delta_E <= 0.0:
            accept = True
        elif T > 0.0:
            accept = np.random.rand() < np.exp(-delta_E / T)
        else:
            accept = False

        if accept:
            positions[i] = new_position
            E_trace.append(E_trace[-1] + delta_E)
        else:
            E_trace.append(E_trace[-1])

    return E_trace