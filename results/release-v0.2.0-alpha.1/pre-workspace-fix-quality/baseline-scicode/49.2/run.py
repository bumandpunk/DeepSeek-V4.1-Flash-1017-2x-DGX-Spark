import numpy as np
import numpy as np

def Nbody_Fij(xin, mass):
    '''This function computes the forces between particles subject to gravity.
    Inputs:
    xin: 2D array of locations for the N particles, N,3 elements where 3 is x,y,z coordinates
    mass: 1D array of particle masses, N elements
    Outputs:
    f: 2D array of forces, on each particle, N,3 elements where 3 is x,y,z coordinates
    '''
    mass = np.asarray(mass, dtype=float)
    x = np.asarray(xin, dtype=float)
    original_shape = x.shape

    if x.ndim == 1:
        N = mass.size
        x = x.reshape((N, 3))
    elif x.ndim == 2:
        N = x.shape[0]
    else:
        raise ValueError("xin must be a 1D array of length 3*N or a 2D array of shape (N, 3)")

    f = np.zeros((N, 3), dtype=float)
    G = 6.67430e-11

    for i in range(N):
        rij = x - x[i]
        r2 = np.sum(rij * rij, axis=1)
        r2[i] = np.inf

        inv_r3 = np.zeros(N, dtype=float)
        mask = r2 > 0.0
        inv_r3[mask] = 1.0 / (r2[mask] * np.sqrt(r2[mask]))

        f[i] = G * mass[i] * np.sum((mass * inv_r3)[:, None] * rij, axis=0)

    if len(original_shape) == 1:
        return f.reshape(-1)
    return f
def Nbody_RHS(uin, mass):
    '''This function computes the right hand side for the evolution equation of a $N$ particle system evolving under gravity. It takes as input the current state $uin$, and mass $mass$.
    Inputs:
    uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    Outputs:
    rhs: a 2D array of velocities and accelerations, N,6 elements; each element is a float; laid out as $vx$, $vy$, $vz$, $ax$, $ay$, $az$.
    '''
    uin = np.asarray(uin, dtype=float)
    mass = np.asarray(mass, dtype=float)

    positions = uin[:, :3]
    velocities = uin[:, 3:6]

    forces = Nbody_Fij(positions, mass)
    accelerations = forces / mass[:, None]

    rhs = np.empty_like(uin)
    rhs[:, :3] = velocities
    rhs[:, 3:6] = accelerations

    return rhs
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('49.2',3,"/data/test_data.h5")

target=targets[0]
uin = np.array([[0.28382835, 0.60608318, 0.94422514, 0.85273554, 0.00225923,
        0.52122603],
       [0.55203763, 0.48537741, 0.76813415, 0.16071675, 0.76456045,
        0.0208098 ]])
mass = np.array([0.17916907, 0.91202765])
assert np.allclose(Nbody_RHS(uin, mass), target)

target=targets[1]
uin = np.array([[0.13521018, 0.11627302, 0.30989758, 0.67145265, 0.47122978,
        0.8161683 ],
       [0.28958678, 0.73312598, 0.70262236, 0.32756948, 0.33464753,
        0.97805808],
       [0.62458211, 0.95031352, 0.76747565, 0.82500925, 0.4066403 ,
        0.45130841]])
mass = np.array([0.26331502, 0.53373939, 0.01457496])
assert np.allclose(Nbody_RHS(uin, mass), target)

target=targets[2]
uin = np.array([[0.40063163, 0.99513816, 0.17756418, 0.9625969 , 0.41925027,
        0.42405245],
       [0.46314887, 0.37372315, 0.4655081 , 0.03516826, 0.08427267,
        0.7325207 ],
       [0.63619999, 0.02790779, 0.30017006, 0.22085252, 0.05501999,
        0.52324607],
       [0.41636966, 0.04821875, 0.57240452, 0.80351665, 0.11360172,
        0.27711721],
       [0.64167809, 0.49273731, 0.50655608, 0.46144026, 0.89474307,
        0.60577769]])
mass = np.array([0.91874701, 0.90071485, 0.03342143, 0.95694934, 0.13720932])
assert np.allclose(Nbody_RHS(uin, mass), target)
