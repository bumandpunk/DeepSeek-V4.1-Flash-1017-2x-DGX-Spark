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
def Nbody_RK4(uin, mass, dt):
    '''This function takes a single step of size $dt$ computing an updated state vector $uout$ given the current state $uin$, $mass$, and timestep $dt$.
    Inputs:
    uin: a 2D array of current positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    dt: time step to use for the evolution; a float
    Outputs:
    uout: a 2D array of final positions and velocities, N,6 elements; each element is a float; laid out like uin
    '''
    uin_array = np.asarray(uin, dtype=float)
    original_shape = uin_array.shape

    if uin_array.ndim == 1:
        mass_array = np.asarray(mass, dtype=float)
        N = mass_array.size
        u = uin_array.reshape((N, 6))
    else:
        u = uin_array

    k1 = Nbody_RHS(u, mass)
    k2 = Nbody_RHS(u + 0.5 * dt * k1, mass)
    k3 = Nbody_RHS(u + 0.5 * dt * k2, mass)
    k4 = Nbody_RHS(u + dt * k3, mass)

    uout = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if len(original_shape) == 1:
        return uout.reshape(-1)

    return uout
def Nbody(uin, mass, dt, t0, t1):
    '''This function runs n-body simulation for N particles interacting via gravity only.
    Inputs:
    uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    dt: time step size; a float
    t0: initial time; a float
    t1: final time; a float
    Outputs:
    out: final positions and velocities of particles; an array of floats laid out like uin
    '''
    uout = np.asarray(uin, dtype=float).copy()
    t = float(t0)
    tf = float(t1)
    h = float(dt)

    if h == 0.0:
        if t == tf:
            return uout
        raise ValueError("dt must be nonzero")

    eps = 1e-12 * max(1.0, abs(tf), abs(t0), abs(h))

    if h > 0.0:
        while t < tf - eps:
            step = min(h, tf - t)
            uout = Nbody_RK4(uout, mass, step)
            t += step
        if abs(tf - t) > eps:
            uout = Nbody_RK4(uout, mass, tf - t)
    else:
        while t > tf + eps:
            step = max(h, tf - t)
            uout = Nbody_RK4(uout, mass, step)
            t += step
        if abs(tf - t) > eps:
            uout = Nbody_RK4(uout, mass, tf - t)

    return uout
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('49.4',4,"/data/test_data.h5")

target=targets[0]
uin = np.array([[0.6033556 , 0.44387186, 0.48004678, 0.88844753, 0.20850049,
        0.94458146],
       [0.07347004, 0.59515246, 0.03115107, 0.66525743, 0.6373855 ,
        0.86246516]])
mass = np.array([0.08463852, 0.31566824])
dt = 0.5
t0 = 0
t1 = 60.
assert np.allclose(Nbody(uin, mass, dt, t0, t1), target)

target=targets[1]
uin = np.array([[0.94163771, 0.44562705, 0.66995763, 0.9243307 , 0.61942347,
        0.32258573],
       [0.31947631, 0.29145669, 0.95741742, 0.40595834, 0.94655582,
        0.85719056],
       [0.68892767, 0.00328797, 0.9001035 , 0.91986487, 0.00590936,
        0.64292615]])
mass = np.array([0.00800334, 0.0104384 , 0.5974588 ])
dt = 0.3
t0 = 30
t1 = 60
assert np.allclose(Nbody(uin, mass, dt, t0, t1), target)

target=targets[2]
uin = np.array([[0.38521698, 0.59574189, 0.61080706, 0.59961919, 0.312284  ,
        0.06987367],
       [0.80073423, 0.91143405, 0.19467463, 0.21060764, 0.37744078,
        0.40284721],
       [0.88517769, 0.20215434, 0.41901483, 0.64795917, 0.51164295,
        0.33383249],
       [0.77847343, 0.13543513, 0.19171077, 0.40258915, 0.78684447,
        0.25380819],
       [0.10948542, 0.93067601, 0.12638728, 0.60500539, 0.29012194,
        0.62652939]])
mass = np.array([0.65947595, 0.06772616, 0.35544292, 0.33472408, 0.17750637])
dt = 0.15
t0 = 0.
t1 = 60.*10.
assert np.allclose(Nbody(uin, mass, dt, t0, t1), target)

target=targets[3]
'''
This function computes the conserved total eenrgy for N particles interacting via gravity only.
Inputs:
uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
mass: an 1D array of particle masses; an array of floats
Outputs:
Etot: the conserved total energy, float
'''
def Nbody_Etot(uin, mass):
    Ekin = 0.5*np.sum(mass * (uin[:,3]**2 + uin[:,4]**2 + uin[:,5]**2))
    # this is horrible, but somewhat close to the naive expression given above
    ggrav = 6.67430e-11
    Epot = 0.
    for i in range(len(mass)):
        for j in range(i+1, len(mass)):
            rij = np.sqrt((uin[i,0] - uin[j,0])**2 + (uin[i,1] - uin[j,1])**2 + (uin[i,2] - uin[j,2])**2)
            Epot += -2. * ggrav * mass[i] * mass[j] / rij
    Etot = Ekin + Epot
    return Etot
# mass of Sun and Earth
mass = np.array([1.98855e30, 5.97219e24])
# initial position and velocity (Sun at origin and at rest, Earth velocity along y axis)
uin = np.array([[0.,0.,0.,
                 0.,0.,0.],
                [1.49598261e11,0.,0.,
                 0.,29.78e3,0.]])
t0 = 0.
t1 = 365.*24.*60.*60. # 1 year
dt = 60.*60.          # 1 hour
Nbody(uin, mass, dt, t0, t1)
Etot0 = Nbody_Etot(uin, mass)
uout = Nbody(uin, mass, dt, t0, t1)
Etot1 = Nbody_Etot(uout, mass)
assert (np.isclose(Etot0, Etot1)) == target
