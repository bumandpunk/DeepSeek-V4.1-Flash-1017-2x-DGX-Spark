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