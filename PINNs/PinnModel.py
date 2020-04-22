from PINNs.PinnLayer import PinnLayer

import tensorflow as tf


class PinnModel(tf.keras.models.Model):

    def __init__(self, simulation_parameters):
        super(PinnModel, self).__init__()

        self.n_buses = simulation_parameters['general']['n_buses']
        self.PinnLayer = PinnLayer(simulation_parameters=simulation_parameters)

        n_data_points = simulation_parameters['data_creation']['n_data_points']
        n_collocation = simulation_parameters['data_creation']['n_collocation']
        n_total = n_data_points + n_collocation

        loss_weights = [n_total / n_data_points,
                        n_total / n_data_points,
                        1]

        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.mean_squared_error,
                     loss_weights=loss_weights)

        self.build(input_shape=[(None, 1), (None, self.n_buses), (None, 1)])

    def call(self, inputs, training=None, mask=None):
        x_time, x_power, x_type = inputs

        network_output, network_output_t, network_output_physics = self.PinnLayer([x_time, x_power])

        loss_network_output_initial = tf.multiply(network_output, x_type)
        loss_network_output_t_initial = tf.multiply(network_output_t, x_type)

        loss_network_output_physics = network_output_physics

        loss_output = (loss_network_output_initial,
                       loss_network_output_t_initial,
                       loss_network_output_physics)

        return loss_output
