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
def MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps):
    curr_positions = init_positions
    curr_velocities = init_velocities

    for _ in range(int(num_steps)):
        curr_positions, curr_velocities = velocity_verlet(
            sigma, epsilon, curr_positions, curr_velocities, dt, m
        )

    return curr_positions, curr_velocities
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('51.4',4,"/data/test_data.h5")

target=targets[0]
import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
m = 1
sigma = 1
epsilon = 1
dt = 0.005
N = 32
num_steps = 1000
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
assert np.allclose(MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps), target)

target=targets[1]
import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
m = 2
sigma = 3
epsilon = 5
dt = 0.01
N = 16
num_steps = 10
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
assert np.allclose(MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps), target)

target=targets[2]
import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
m = 1
sigma = 1
epsilon = 1
dt = 0.02
N = 8
num_steps = 20
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
assert np.allclose(MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps), target)

target=targets[3]
import itertools
def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return np.array(position)
def kinetic_energy(velocities, m):
    return 0.5 * m * np.sum(velocities**2)
def potential_energy(positions, epsilon, sigma):
    V = 0.0
    N = len(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            V += 4 * epsilon * ((sigma / r_ij)**12 - (sigma / r_ij)**6)
    return V
def total_energy(positions, velocities, m, epsilon, sigma):
    K = kinetic_energy(velocities, m)
    V = potential_energy(positions, epsilon, sigma)
    return K + V
def linear_momentum(velocities, m):
    return m * np.sum(velocities, axis=0)
def angular_momentum(positions, velocities, m):
    L = np.zeros(3)
    for i in range(len(positions)):
        L += np.cross(positions[i], m * velocities[i])
    return L
m = 1
sigma = 1
epsilon = 1
dt = 0.0025
N = 32
num_steps = 1000
init_positions = initialize_fcc(N)
init_velocities = np.zeros(init_positions.shape)
positions,velocities = MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps)
# Initial values
initial_energy = total_energy(init_positions, init_velocities, m, epsilon, sigma)
initial_linear_momentum = linear_momentum(init_velocities, m)
initial_angular_momentum = angular_momentum(init_positions, init_velocities, m)
# Final values (after simulation)
final_energy = total_energy(positions, velocities, m, epsilon, sigma)
final_linear_momentum = linear_momentum(velocities, m)
final_angular_momentum = angular_momentum(positions, velocities, m)
# Checking conservation
assert (np.isclose(final_energy,initial_energy), np.allclose(final_linear_momentum, initial_linear_momentum),
        np.allclose(final_angular_momentum, initial_angular_momentum)) == target
