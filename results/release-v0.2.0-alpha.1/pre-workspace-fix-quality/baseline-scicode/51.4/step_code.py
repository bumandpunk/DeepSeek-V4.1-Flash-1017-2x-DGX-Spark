def MD(init_positions, init_velocities, sigma, epsilon, m, dt, num_steps):
    curr_positions = init_positions
    curr_velocities = init_velocities

    for _ in range(int(num_steps)):
        curr_positions, curr_velocities = velocity_verlet(
            sigma, epsilon, curr_positions, curr_velocities, dt, m
        )

    return curr_positions, curr_velocities