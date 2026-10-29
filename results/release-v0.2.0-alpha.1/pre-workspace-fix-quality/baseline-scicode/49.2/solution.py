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