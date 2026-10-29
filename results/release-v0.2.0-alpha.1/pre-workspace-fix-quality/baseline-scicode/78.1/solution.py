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