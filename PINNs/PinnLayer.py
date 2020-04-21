import tensorflow as tf


class DenseHelperModel(tf.keras.models.Model):
    """
    This constitutes the core neural network with the PINNs model. It outputs the angle for each generator based on
    the time input.
    """

    def __init__(self, simulation_parameters):

        super(DenseHelperModel, self).__init__()

        tf.random.set_seed(simulation_parameters['training']['tensorflow_seed'])

        self.n_buses = simulation_parameters['general']['n_buses']
        self.neurons_in_hidden_layers = simulation_parameters['training']['neurons_in_hidden_layers']

        self.hidden_layers = []
        for n_neurons in self.neurons_in_hidden_layers:
            self.hidden_layers.append(tf.keras.layers.Dense(units=n_neurons,
                                                            activation=tf.keras.activations.tanh,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                                            bias_initializer=tf.keras.initializers.zeros))

        self.dense_output_layer = tf.keras.layers.Dense(units=self.n_buses,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                                        bias_initializer=tf.keras.initializers.zeros)

        self.normalise_input_layer = NormaliseInputLayer(simulation_parameters=simulation_parameters)

    def call(self, inputs, training=None, mask=None):

        input_time = inputs

        hidden_layer_input = self.normalise_input_layer(input_time)

        for layer in self.hidden_layers:
            hidden_layer_input = layer(hidden_layer_input)

        network_output = self.dense_output_layer(hidden_layer_input)

        return network_output


class NormaliseInputLayer(tf.keras.layers.Layer):
    """
    Normalise the time input from [0, t_max] to [-1, 1]
    """

    def __init__(self, simulation_parameters):
        super(NormaliseInputLayer, self).__init__()

        # [[t_min], [t_max]] -> [2, 1]
        self.t_max = simulation_parameters['general']['t_max']
        self.t_min = 0

    def call(self, inputs, **kwargs):
        time_input = inputs
        time_normalised = 2.0 * (time_input - self.t_min) / (self.t_max - self.t_min) - 1.0

        return time_normalised


def get_gradients_in_time(model, features, bus):
    with tf.GradientTape(watch_accessed_variables=False,
                         persistent=False) as grad_tt:
        grad_tt.watch(features)
        with tf.GradientTape(watch_accessed_variables=False,
                             persistent=False) as grad_t:
            grad_t.watch(features)
            network_output_single_bus = model(features)[:, bus:bus + 1]

        network_output_t_single_bus = grad_t.gradient(network_output_single_bus,
                                                      features,
                                                      unconnected_gradients='zero')
    network_output_tt_single_bus = grad_tt.gradient(network_output_t_single_bus,
                                                    features,
                                                    unconnected_gradients='zero')

    return network_output_single_bus, network_output_t_single_bus, network_output_tt_single_bus


class PinnLayer(tf.keras.layers.Layer):
    """
    This layer includes the prediction
    """

    def __init__(self, simulation_parameters):
        super(PinnLayer, self).__init__()

        self.n_buses = simulation_parameters['general']['n_buses']
        self.bus_with_inertia = simulation_parameters['general']['bus_with_inertia']
        self.lambda_m_true = simulation_parameters['true_system']['lambda_m']
        self.lambda_d_true = simulation_parameters['true_system']['lambda_d']

        self.lambda_m = tf.Variable(tf.ones(shape=(1, self.n_buses)),
                                    trainable=True,
                                    name='lambda_m',
                                    dtype=tf.float32)

        self.lambda_d = tf.Variable(tf.ones(shape=(1, self.n_buses)),
                                    trainable=True,
                                    name='lambda_d',
                                    dtype=tf.float32)

        self.lambda_b = tf.Variable(simulation_parameters['true_system']['lambda_b'],
                                    trainable=False,
                                    name='lambda_b',
                                    dtype=tf.float32)

        self.DenseLayers = DenseHelperModel(simulation_parameters=simulation_parameters)

    def print_relative_error(self):

        inertia_error = []
        damping_error = []
        for bus in range(self.n_buses):
            if self.bus_with_inertia[0, bus]:
                inertia_error.append(
                    abs(self.lambda_m[0, bus] - self.lambda_m_true[0, bus]) / self.lambda_m_true[0, bus] * 100)
            else:
                inertia_error.append(tf.constant(0.0))
            damping_error.append(
                abs(self.lambda_d[0, bus] - self.lambda_d_true[0, bus]) / self.lambda_d_true[0, bus] * 100)
        inertia_error = tf.stack(inertia_error)
        damping_error = tf.stack(damping_error)
        print(f'Relative error of m in %: {inertia_error.numpy()}')
        print(f'Relative error of d in %: {damping_error.numpy()}')

    def call(self, inputs, **kwargs):
        input_time, input_power = inputs

        list_network_output = []
        list_network_output_t = []
        list_network_output_tt = []

        for bus in range(self.n_buses):
            network_output_single_bus, network_output_t_single_bus, network_output_tt_single_bus = get_gradients_in_time(model=self.DenseLayers,
                                                                 features=input_time,
                                                                 bus=bus)
            list_network_output.append(network_output_single_bus)
            list_network_output_t.append(network_output_t_single_bus)
            list_network_output_tt.append(network_output_tt_single_bus)

        network_output = tf.concat(list_network_output, axis=1)
        network_output_t = tf.concat(list_network_output_t, axis=1)
        network_output_tt = tf.concat(list_network_output_tt, axis=1)

        delta_i = tf.repeat(input=tf.reshape(network_output, [-1, self.n_buses, 1]),
                            repeats=self.n_buses,
                            axis=2)

        if self.n_buses == 1:
            delta_j = delta_i * 0
        else:
            delta_j = tf.repeat(input=tf.reshape(network_output, [-1, 1, self.n_buses]),
                                repeats=self.n_buses,
                                axis=1)

        connectivity_matrix = self.lambda_b * tf.math.sin(delta_i - delta_j)
        connectivity_vector = tf.reduce_sum(connectivity_matrix, axis=2)

        network_output_physics = (self.lambda_m * self.bus_with_inertia * network_output_tt +
                                  self.lambda_d * network_output_t +
                                  connectivity_vector - input_power)

        return network_output, network_output_t, network_output_physics
