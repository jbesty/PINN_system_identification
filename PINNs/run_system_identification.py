import numpy as np
import time

from PINNs.create_example_parameters import create_example_parameters
from PINNs.create_data import create_data
from PINNs.PinnModel import PinnModel


def run_system_identification():

    # load or create a file with all simulation parameters such that a simulation is repeatable
    # to illustrate the working principle, examples for 1 and 4 buses are implemented
    simulation_parameters = create_example_parameters(n_buses=4)

    # at this point the training data are provided
    # here we simulate a dataset based on the previously defined simulation parameters
    x_training, y_training = create_data(simulation_parameters=simulation_parameters)

    # creating the model including building it and setting the options for the optimiser, the loss function and the
    # loss weights --> see PinnModel.py
    model = PinnModel(simulation_parameters=simulation_parameters)

    np.set_printoptions(precision=3)
    print('Starting training')
    total_start_time = time.time()

    for n_epochs, batch_size in zip(simulation_parameters['training']['epoch_schedule'],
                                    simulation_parameters['training']['batching_schedule']):

        epoch_start_time = time.time()
        model.fit(x_training,
                  y_training,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=True)
        epoch_end_time = time.time()

        print(f'Trained for {n_epochs} epochs with batch size {batch_size} '
              f'in {epoch_end_time - epoch_start_time:.2f} seconds.')

        model.PinnLayer.print_relative_error()

    total_end_time = time.time()
    print(f'Total training time: {total_end_time - total_start_time:.1f} seconds')


if __name__ == "__main__":
    run_system_identification()
