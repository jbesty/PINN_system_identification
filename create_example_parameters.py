import numpy as np


def create_example_parameters(n_buses: int):
    """
    creates a basic set of parameters that are used in the following processes:
    * data creation if measurements are to be simulated
    * setting up the neural network model
    * training procedure

    :param n_buses: integer number of buses in the system
    :return: simulation_parameters: dictonary that holds all parameters
    """

    # -----------------------------------------------------------------------------------------------
    # underlying parameters of the power system
    # primarily for data creation when no measurements are provided
    # lambda_m: inertia of the component at each bus (0 for load buses) -> shape [1, n_buses]
    # lambda_d: damping of the component at each bus -> shape [1, n_buses]
    # lambda_b: B-matrix of the power system -> shape [n_buses, n_buses]
    # power_set_points: vector of constant power disturbance -> shape [1, n_buses] 
    # delta_initial: rotor angle at each bus at t = 0 -> shape [1, 1, n_buses]
    # omega_initial: frequency at each bus at t = 0 -> shape [1, 1, n_buses]
    # -----------------------------------------------------------------------------------------------

    if n_buses == 1:
        lambda_m = np.array([0.4]).reshape((1, n_buses))
        lambda_d = np.array([0.15]).reshape((1, n_buses))
        lambda_b = np.array([0.2])

        power_set_points = np.array([0.1])
        delta_initial = np.array([0.0])
        omega_initial = np.array([0.0])

    elif n_buses == 4:
        lambda_m = np.array([0.3, 0.2, 0.0, 0.0]).reshape((1, n_buses))
        lambda_d = np.array([0.15, 0.3, 0.25, 0.2]).reshape((1, n_buses))
        lambda_b = np.array([[1.0, 0.0, 0.5, 1.2],
                             [0.0, 1.0, 1.4, 0.8],
                             [0.5, 1.4, 1.0, 0.1],
                             [1.2, 0.8, 0.1, 1.0]])

        power_set_points = np.array([0.1, 0.2, -0.2, -0.1])
        delta_initial = np.array([0.0, 0.0, 0.0, 0.0])
        omega_initial = np.array([0.0, 0.0, 0.0, 0.0])

    else:
        raise Exception("The chosen N-bus-system is not implemented")

    true_system_parameters = {'lambda_m': lambda_m,
                              'lambda_d': lambda_d,
                              'lambda_b': lambda_b,
                              'power_set_point': power_set_points,
                              'delta_initial': delta_initial,
                              'omega_initial': omega_initial}

    # -----------------------------------------------------------------------------------------------
    # general parameters of the power system that are assumed to be known in the identification process
    # n_buses: integer number of buses in the system
    # time_window: [t_min, t_max] -> shape [2, 1]
    # bus_with_inertia: boolean for each bus to indicate whether inertia is present or not -> shape [1, n_buses]
    # -----------------------------------------------------------------------------------------------
    t_max = 2
    bus_with_inertia = lambda_m > 0

    general_parameters = {'n_buses': n_buses,
                          't_max': t_max,
                          'bus_with_inertia': bus_with_inertia}

    # -----------------------------------------------------------------------------------------------
    # parameters for the training data creation 
    # n_data_points: number of data points where measurements are present
    # n_collocation_points: number of points where the physics are evaluated at (additional to the data points)
    # noise_level: standard deviation of the zero-mean Gaussian measurement noise
    # noise_seed: seed for the addition of the random noise
    # -----------------------------------------------------------------------------------------------
    n_data_points = 21
    n_collocation_points = 80

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_collocation': n_collocation_points,
                                'noise_level': 0.01,
                                'noise_seed': 1}

    # -----------------------------------------------------------------------------------------------
    # parameters for the scheduled training process and the network architecture
    # epoch_schedule: number of epochs per batch size
    # batching_schedule: batch size
    # neurons_in_hidden_layers: number of neurons for each hidden layer
    # tensorflow_seed: seed for the random initalisation of the networks weight matrices
    # -----------------------------------------------------------------------------------------------
    n_total = n_data_points + n_collocation_points
    epoch_schedule = [500, 1000, 2000, 5000, 10000]
    # epoch_schedule = [1, 1000, 2000, 5000, 10000]

    batching_schedule = [int(np.ceil(n_total / 20)),
                         int(np.ceil(n_total / 10)),
                         int(np.ceil(n_total / 5)),
                         int(np.ceil(n_total / 2)),
                         int(n_total)]

    training_parameters = {'epoch_schedule': epoch_schedule,
                           'batching_schedule': batching_schedule,
                           'neurons_in_hidden_layers': [30, 30],
                           'tensorflow_seed': 1}

    # -----------------------------------------------------------------------------------------------
    # combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters,
                             'training': training_parameters}

    return simulation_parameters
