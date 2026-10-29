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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('78.1',3,"/data/test_data.h5")

target=targets[0]
def test_case_1():
    state = np.array([0.1, 0.0])  # initial theta and omega
    t = 0.0  # initial time
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.0  # no damping
    A = 0.0  # no driving force
    alpha = 0.0  # driving frequency (irrelevant since A=0)
    expected_output = np.array([0.0, - (g / L) * np.sin(0.1)])  # [dtheta_dt, domega_dt]
    output = pendulum_derivs(state, t, g, L, beta, A, alpha)
    assert np.allclose(output, expected_output), "Test Case 1 Failed"
# Run test cases
test_case_1()

target=targets[1]
def test_case_2():
    state = np.array([0.1, 0.2])  # initial theta and omega
    t = 0.0  # initial time
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.1  # damping coefficient
    A = 0.0  # no driving force
    alpha = 0.0  # driving frequency (irrelevant since A=0)
    expected_output = np.array([0.2, - (g / L) * np.sin(0.1) - 0.1 * 0.2])  # [dtheta_dt, domega_dt]
    output = pendulum_derivs(state, t, g, L, beta, A, alpha)
    assert np.allclose(output, expected_output), "Test Case 2 Failed"
test_case_2()

target=targets[2]
def test_case_3():
    state = np.array([0.1, 0.2])  # initial theta and omega
    t = 1.0  # time
    g = 9.81  # gravity
    L = 1.0  # pendulum length
    beta = 0.1  # damping coefficient
    A = 1.0  # amplitude of driving force
    alpha = 2.0  # driving frequency
    expected_output = np.array([0.2, - (g / L) * np.sin(0.1) - 0.1 * 0.2 + 1.0 * np.cos(2.0 * 1.0)])  # [dtheta_dt, domega_dt]
    output = pendulum_derivs(state, t, g, L, beta, A, alpha)
    assert np.allclose(output, expected_output), "Test Case 3 Failed"
test_case_3()
