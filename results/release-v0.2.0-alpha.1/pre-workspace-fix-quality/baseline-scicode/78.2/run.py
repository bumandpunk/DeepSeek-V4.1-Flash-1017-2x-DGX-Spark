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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('78.2',3,"/data/test_data.h5")

target=targets[0]
def test_case_1():
    state = np.array([0.1, 0.0])  # initial theta and omega
    t0 = 0.0  # initial time
    dt = 0.01  # time step
    n = 1000  # number of steps
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.0  # no damping
    A = 0.0  # no driving force
    alpha = 0.0  # driving frequency (irrelevant since A=0)
    trajectory = runge_kutta_4th_order(pendulum_derivs, state, t0, dt, n, g, L, beta, A, alpha)
    final_state = trajectory[-1]
    expected_theta = state[0] * np.cos(np.sqrt(g / L) * n * dt)  # Simple harmonic motion solution
    assert np.isclose(final_state[0], expected_theta, atol=0.1), "Test Case 1 Failed"
# Run test cases
test_case_1()

target=targets[1]
def test_case_2():
    state = np.array([0.1, 0.0])  # initial theta and omega
    t0 = 0.0  # initial time
    dt = 0.01  # time step
    n = 1000  # number of steps
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.1  # damping coefficient
    A = 0.0  # no driving force
    alpha = 0.0  # driving frequency (irrelevant since A=0)
    trajectory = runge_kutta_4th_order(pendulum_derivs, state, t0, dt, n, g, L, beta, A, alpha)
    final_state = trajectory[-1]
    assert final_state[0] < state[0], "Test Case 2 Failed"  # Expecting damping to reduce theta
test_case_2()

target=targets[2]
def test_case_3():
    state = np.array([0.1, 0.0])  # initial theta and omega
    t0 = 0.0  # initial time
    dt = 0.01  # time step
    n = 1000  # number of steps
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.1  # damping coefficient
    A = 1.0  # amplitude of driving force
    alpha = 2.0  # driving frequency
    trajectory = runge_kutta_4th_order(pendulum_derivs, state, t0, dt, n, g, L, beta, A, alpha)
    final_state = trajectory[-1]
    assert not np.isnan(final_state).any(), "Test Case 3 Failed"  # Check if the result is a valid number
test_case_3()
