import numpy as np
def f_ij(sigma, epsilon, r):
    '''This function computes the force between two particles interacting through Lennard Jones Potantial
    Inputs:
    sigma: the distance at which potential reaches zero, float
    epsilon: potential well depth, float
    r: 3D displacement between two particles, 1d array of float with shape (3,)
    Outputs:
    f: force, 1d array of float with shape (3,)
    '''
    r_norm = np.linalg.norm(r)
    sr6 = (sigma / r_norm) ** 6
    sr12 = sr6 ** 2
    factor = 24.0 * epsilon * (2.0 * sr12 - sr6) / (r_norm ** 2)
    f = factor * r
    return f
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
def velocity_verlet(sigma, epsilon, positions, velocities, dt, m):
    '''This function runs Velocity Verlet algorithm to integrate the positions and velocities of atoms interacting through
    Lennard Jones Potential forward for one time step according to Newton's Second Law.
    Inputs:
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float 
    positions: current positions of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    velocities: current velocities of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    dt: time step size, float
    m: mass,float
    Outputs:
    new_positions: new positions of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    new_velocities: new velocities of all atoms, 2D array of floats with shape (N,3) where N is the number of atoms, 3 is x,y,z coordinate
    '''
    accelerations = aggregate_forces(sigma, epsilon, positions) / m
    half_velocities = velocities + 0.5 * accelerations * dt

    new_positions = positions + half_velocities * dt

    new_accelerations = aggregate_forces(sigma, epsilon, new_positions) / m
    new_velocities = half_velocities + 0.5 * new_accelerations * dt

    return new_positions, new_velocities
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('51.3',3,"/data/test_data.h5")

target=targets[0]
sigma = 1
epsilon = 1
positions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])
velocities = np.array([[0.78759462, 0.55733813, 0.97321828],
       [0.67351598, 0.74988225, 0.2430374 ]])
dt = 0.005
m = 1
assert np.allclose(velocity_verlet(sigma,epsilon,positions,velocities,dt,m), target)

target=targets[1]
sigma = 2
epsilon = 3
positions = np.array([[2.15569639, 3.33938018, 0.1682584 ],[0.20055338, 5.56353376, 1.9202491], [0.61941565, 0.19380897, 0.20261979 ]])
velocities = np.array([[0.90594996, 0.96932907, 0.57531663],
       [0.80310603, 0.96969681, 0.1771265 ],
       [0.87580384, 0.42642859, 0.40785355]])
dt = 0.01
m = 3
assert np.allclose(velocity_verlet(sigma,epsilon,positions,velocities,dt,m), target)

target=targets[2]
sigma = 5
epsilon = 2
positions = np.array([[5.62726631, 5.3077771 , 7.29719649],
       [7.25031287, 7.58926428, 3.71262908],
       [8.7866416 , 3.73724676, 9.22676027],
       [7.89096788, 5.3872004 , 7.95350911],
       [6.068183  , 3.55807037, 2.7965242 ]])
velocities = np.array([[0.21901831, 0.03442682, 0.56648053],
       [0.70408383, 0.89261961, 0.08422311],
       [0.43673835, 0.28973691, 0.74121155],
       [0.06514819, 0.59291869, 0.7190472 ],
       [0.40667563, 0.28100076, 0.08050432]])
dt = 0.2
m = 5
assert np.allclose(velocity_verlet(sigma,epsilon,positions,velocities,dt,m), target)
