# simple_function_test

## Dependencies
* Python 3.6
* Tensorflow
* Keras
* Matplotlib

## Description
* /networks
  * /dataset
    - dataset_generator.py generates data given a function object, splitting into training and testing
	- dataset_files.py save and load datasets from and to files
  * /models
    - initializers.py: Custom initializers for neural network model parameters using weight initialization procedures from: https://arxiv.org/abs/1704.08863
	- metrics.py: Mean sqaured error metric
	- model_base.py: Abstract Keras model class with training procedure
	- simple_dense.py: Keras model implementation of basic multi-layer dense network
  * /visualizers
    - simple_dense_visualizer.py: Visualization function for plotting data during training
* /simple_funcs
  - func_composition.py: Composes functions based on a single neural network layer into a single function object similar to a random dense neural network
  - function_base.py: Abstract base class for function objectives
  - step_funcs.py: Creates a function object for a step function of varying complexity, i.e. simple functions
- function_params.py: parameters used to create function object at runtime
- model_params.py: parameters used for building neural network model at runtime
- main.py: running a training session given a model and a dataset
- generate_test_data.py: generate dataset files from a function object
- generate_stats_plots.py: Generate statistics and plots from a training session
- generate_qsubs.py: Used for generating qsub commands for running many tests on the EPFL cluster
- run_from_qsub.py: Used by the qsub commands to run training on an EPFL cluster node
