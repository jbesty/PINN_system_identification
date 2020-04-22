# Physics-Informed Neural Networks for Non-linear System Identification applied to Power System Dynamics

Code related to the submission of
- Jochen Stiasny and George S. Misyris and Spyros Chatzivasileiadis. "[Physics-Informed Neural Networks for Non-linear System Identification applied to Power System Dynamics](https://arxiv.org/abs/2004.04026)." arXiv preprint arXiv:2004.04026 (2020).


##  Code structure
The code is structured in the following way:
- `run_system_identification.py` contains the entire workflow
- `create_example_parameters.py` creates a dictionary with all relevant information about the system and the training which are used throughout the process
- `create_data.py` to showcase the method in the absence of measurement data, this function creates the training data
- `ode_solver.py` a simple ode-solver used in `create_data.py`
- `PinnModel.py` the network model that inherits from the class `tensorflow.keras.models.Model`
- `PinnLayer.py` the layer (inheriting from `tensorflow.keras.layers.Layer`) that combines the dense neural network with the automatic differentiation

## Citation

    @misc{stiasny2020physicsinformed,
        title={Physics-Informed Neural Networks for Non-linear System Identification applied to Power System Dynamics},
        author={Jochen Stiasny and George S. Misyris and Spyros Chatzivasileiadis},
        year={2020},
        eprint={2004.04026},
        archivePrefix={arXiv},
        primaryClass={eess.SY}
    }
 
 ## Related work
 
 The concept of PINNs was introduced by Raissi et al. (https://maziarraissi.github.io/PINNs/) and adapted to power systems by Misyris et al. (https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems). The presented code is inspired by these two sources. 
