import numpy as np
from PINNs.ode_solver import solve_ode


def create_data(simulation_parameters):
    np.random.seed(simulation_parameters['data_creation']['noise_seed'])

    n_buses = simulation_parameters['general']['n_buses']
    n_collocation = simulation_parameters['data_creation']['n_collocation']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    n_total = n_data_points + n_collocation

    t_max = simulation_parameters['general']['t_max']
    x_time = np.linspace(0, t_max, n_data_points).reshape((-1, 1))

    power_initial = simulation_parameters['true_system']['power_set_point'].reshape((1, n_buses))

    x_power = power_initial.repeat(repeats=n_collocation + n_data_points, axis=0)

    results = solve_ode(x_time[:, 0], simulation_parameters)

    if n_collocation > 0:
        x_time_collocation = np.linspace(0, t_max, n_total)
        steps_to_delete = np.s_[::int(np.ceil(n_total / n_data_points))]
        x_time_collocation = np.delete(x_time_collocation, steps_to_delete).reshape((-1, 1))
    else:
        x_time_collocation = np.zeros((0, 1))

    x_type_collocation = np.zeros((n_collocation, 1))

    x_type_data = np.ones((n_data_points, 1))

    y_delta_data = results[:, 0, :] * (1 + simulation_parameters['data_creation']['noise_level'] *
                                           np.random.randn(x_time.shape[0], n_buses))
    y_omega_data = results[:, 1, :] * (1 + simulation_parameters['data_creation']['noise_level'] *
                                           np.random.randn(x_time.shape[0], n_buses))

    y_collocation = np.zeros((n_collocation, n_buses))

    y_delta = np.concatenate([y_delta_data, y_collocation], axis=0)
    y_omega = np.concatenate([y_omega_data, y_collocation], axis=0)

    x_training = [np.concatenate([x_time, x_time_collocation], axis=0),
                  x_power,
                  np.concatenate([x_type_data, x_type_collocation], axis=0)]

    y_training = [y_delta, y_omega, np.zeros((n_data_points + n_collocation, 1))]

    return x_training, y_training
