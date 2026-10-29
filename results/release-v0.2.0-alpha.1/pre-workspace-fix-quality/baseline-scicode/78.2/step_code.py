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