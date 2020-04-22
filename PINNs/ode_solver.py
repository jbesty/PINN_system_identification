import numpy as np
from scipy import integrate


def ode_right_hand_side(t, state_variable, n_buses, lambda_m, lambda_d, lambda_b, power):
    """
    system of first order ordinary differential equations
    :param t: variable if ode depends on t
    :param state_variable: state vector consisting of delta_i and omega_i for i in 1:n_buses
    :param n_buses: number of buses
    :param lambda_m: inertia at each bus
    :param lambda_d: damping coefficient at each bus
    :param lambda_b: bus susceptance matrix
    :param power: power injection or retrieval at each bus
    :return: updated state variable
    """
    # split the state variable into delta and omega
    state_delta = state_variable[:n_buses].reshape((-1, 1))
    state_omega = state_variable[n_buses:].reshape((-1, 1))

    # computing the non-linear term in the swing equation sum_j (B_ij sin(delta_i - delta_j))
    delta_i = np.repeat(state_delta, repeats=n_buses, axis=1)
    if n_buses == 1:
        delta_j = np.zeros(shape=delta_i.shape)
    else:
        delta_j = np.repeat(state_delta.reshape((1, -1)), repeats=n_buses, axis=0)

    delta_ij = np.sin(delta_i - delta_j)
    connectivity_vector = np.sum(np.multiply(lambda_b, delta_ij), axis=1).reshape((-1, 1))

    # update states
    state_delta_new = np.zeros(shape=state_delta.shape)
    state_omega_new = np.zeros(shape=state_omega.shape)

    for bus in range(n_buses):
        if lambda_m[bus] > 0:
            state_delta_new[bus] = state_omega[bus]
            state_omega_new[bus] = 1 / lambda_m[bus] * (
                        power[bus] - lambda_d[bus] * state_omega[bus] - connectivity_vector[bus])
        elif lambda_d[bus] > 0:
            state_delta_new[bus] = 1 / lambda_d[bus] * (power[bus] - connectivity_vector[bus])
            state_omega_new[bus] = 0
        else:
            state_delta_new[bus] = 0
            state_omega_new[bus] = 0

    return np.concatenate([state_delta_new[:, 0],
                           state_omega_new[:, 0]],
                          axis=0)


def evaluate_algebraic_equations(state_evolution, n_buses, lambda_m, lambda_d, lambda_b, power):
    """
    evaluate all states (here omega_i for buses with lambda_m == 0) described by algebraic equations
    :param state_evolution: state variables over time of shape [n_timesteps, states, n_buses]
        where states represents delta and omega
    :param n_buses: number of buses
    :param lambda_m: inertia at each bus
    :param lambda_d: damping coefficient at each bus
    :param lambda_b: bus susceptance matrix
    :param power: power injection or retrieval at each bus
    :return: updated state_evolution
    """

    # computing the non-linear term in the swing equation Sum_j [B_ij sin(delta_i - delta_j)]
    lambda_b = lambda_b.reshape(1, n_buses, n_buses)
    delta_i = np.repeat(state_evolution[:, 0, :].reshape((-1, n_buses, 1)),
                        repeats=n_buses,
                        axis=2)

    if n_buses == 1:
        delta_j = delta_i * 0
    else:
        delta_j = np.repeat(state_evolution[:, 0, :].reshape((-1, 1, n_buses)),
                            repeats=n_buses,
                            axis=1)

    connectivity_matrix = lambda_b * np.sin(delta_i - delta_j)
    connectivity_vector = np.sum(connectivity_matrix, axis=2)

    # update states for all time steps at once
    for bus in range(n_buses):
        if lambda_m[bus] > 0:
            pass
        elif lambda_d[bus] > 0:
            state_evolution[:, 1, bus] = 1 / lambda_d[bus] * (power[bus] - connectivity_vector[:, bus])
        else:
            # TODO: evaluate algebraic equations for buses with non-frequency dependent load
            pass

    return state_evolution


def solve_ode(x_time, simulation_parameters):
    # check for dimensions of all input variables
    n_buses = simulation_parameters['general']['n_buses']
    lambda_m = simulation_parameters['true_system']['lambda_m'].reshape((-1, 1))
    lambda_d = simulation_parameters['true_system']['lambda_d'].reshape((-1, 1))
    lambda_b = simulation_parameters['true_system']['lambda_b']
    t_span = np.array([0, simulation_parameters['general']['t_max']])

    power = simulation_parameters['true_system']['power_set_point'].reshape((-1, 1))
    delta_initial = simulation_parameters['true_system']['delta_initial'].reshape((-1, 1))
    omega_initial = simulation_parameters['true_system']['omega_initial'].reshape((-1, 1))
    states_initial = np.concatenate([delta_initial, omega_initial], axis=0)[:, 0]

    ode_solution = integrate.solve_ivp(ode_right_hand_side,
                                       t_span=t_span,
                                       y0=states_initial,
                                       args=[n_buses, lambda_m, lambda_d, lambda_b, power],
                                       t_eval=x_time)

    state_results = np.transpose(ode_solution.y).reshape((-1, 2, n_buses))

    state_results_complete = evaluate_algebraic_equations(state_results, n_buses, lambda_m, lambda_d, lambda_b, power)

    return state_results_complete
