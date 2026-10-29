import numpy as np
import time
def pendulum_derivs(state, t, g, L, beta, A, alpha):
    theta = state[0]
    omega = state[1]

    dtheta_dt = omega
    domega_dt = (
        -(g / L) * np.sin(theta)
        - (beta / L) * omega
        + (A / (L ** 2)) * np.cos(alpha * t)
    )

    state_matrix = np.array([dtheta_dt, domega_dt])
    return state_matrix
def runge_kutta_4th_order(f, state, t0, dt, n, g, L, beta, A, alpha):
    '''Run the RK4 integrator to solve the pendulum motion.
    Inputs:
    f: Derivatives function, which in general can be any ODE. In this context, it is defined as the pendulum_derivs function.
    state: Initial state vector [theta(t0), omega(t0)], 1D numpy array of length 2.
    t0: Initial time, float.
    dt: Time step, float.
    n: Number of steps, int.
    g: Acceleration due to gravity, float.
    L: Length of the pendulum, float.
    beta: Damping coefficient, float.
    A: Amplitude of driving force, float.
    alpha: Frequency of driving force, float.
    Outputs:
    trajectory: Array of state vectors, 2D numpy array of shape (n+1, 2).
    '''

    state = np.asarray(state, dtype=float)
    trajectory = np.zeros((n + 1, 2), dtype=float)
    trajectory[0] = state

    for i in range(n):
        t = t0 + i * dt
        y = trajectory[i]

        k1 = f(y, t, g, L, beta, A, alpha)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, g, L, beta, A, alpha)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, g, L, beta, A, alpha)
        k4 = f(y + dt * k3, t + dt, g, L, beta, A, alpha)

        trajectory[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return trajectory
def pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps):
    '''Analyze a damped, driven pendulum system under various conditions.
    Inputs:
    g: Acceleration due to gravity, float.
    L: Length of the pendulum, float.
    beta: Damping coefficient, float.
    A: Amplitude of driving force, float.
    alpha: Frequency of driving force, float.
    initial_state: Initial state vector [theta, omega], list of floats.
    t0: Initial time, float.
    tf: Final time, float.
    min_dt: Smallest timestep, float.
    max_dt: Largest timestep, float.
    num_timesteps: Number of timesteps to generate, int.
    Outputs:
    optimized_trajectory: numpy array of floats, The trajectory of the pendulum system using the optimized timestep.
    '''
    initial_state = np.asarray(initial_state, dtype=float)
    duration = tf - t0

    if duration <= 0.0:
        return np.array([initial_state], dtype=float)

    if num_timesteps < 1:
        num_timesteps = 1

    if min_dt <= 0.0 or max_dt <= 0.0:
        raise ValueError("min_dt and max_dt must be positive.")

    dt_candidates = np.linspace(min_dt, max_dt, num_timesteps)

    best_metric = np.inf
    best_dt = dt_candidates[0]

    for dt in dt_candidates:
        if dt <= 0.0:
            continue

        n_coarse = max(1, int(np.ceil(duration / dt)))
        h_coarse = duration / n_coarse

        start_time = time.perf_counter()

        coarse_trajectory = runge_kutta_4th_order(
            pendulum_derivs,
            initial_state,
            t0,
            h_coarse,
            n_coarse,
            g,
            L,
            beta,
            A,
            alpha
        )

        n_fine = 2 * n_coarse
        h_fine = h_coarse / 2.0

        fine_trajectory = runge_kutta_4th_order(
            pendulum_derivs,
            initial_state,
            t0,
            h_fine,
            n_fine,
            g,
            L,
            beta,
            A,
            alpha
        )

        elapsed_time = time.perf_counter() - start_time

        difference = fine_trajectory[-1] - coarse_trajectory[-1]
        gte = np.linalg.norm(difference) / 15.0

        if elapsed_time <= 0.0:
            elapsed_time = np.finfo(float).eps

        metric = gte * np.sqrt(elapsed_time)

        if metric < best_metric:
            best_metric = metric
            best_dt = dt

    n_optimized = max(1, int(np.ceil(duration / best_dt)))
    optimized_dt = duration / n_optimized

    optimized_trajectory = runge_kutta_4th_order(
        pendulum_derivs,
        initial_state,
        t0,
        optimized_dt,
        n_optimized,
        g,
        L,
        beta,
        A,
        alpha
    )

    return optimized_trajectory
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('78.3',3,"/data/test_data.h5")

target=targets[0]
g = 9.81
L = 0.1
beta = 0.1
A = 0.0
alpha = 0.0
initial_state = [0.1, 0.0]
t0 = 0.0
tf = 10.0
min_dt = 0.001
max_dt = 10
num_timesteps = 2
assert np.allclose(pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps), target)

target=targets[1]
g = 9.81
L = 0.5
beta = 0.2
A = 1.0
alpha = 0.5
initial_state = [0.5, 0.1]
t0 = 0.0
tf = 20.0
min_dt = 0.001
max_dt = 10
num_timesteps = 2
assert np.allclose(pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps), target)

target=targets[2]
g = 9.81
L = 0.3
beta = 1.0
A = 0.2
alpha = 1.0
initial_state = [1.0, 0.0]
t0 = 0.0
tf = 15.0
min_dt = 0.001
max_dt = 10
num_timesteps = 2
assert np.allclose(pendulum_analysis(g, L, beta, A, alpha, initial_state, t0, tf, min_dt, max_dt, num_timesteps), target)
