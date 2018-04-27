import math
import numpy as np
import sys
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from keras.optimizers import Adam, SGD, RMSprop, Nadam

from model_params import *
from function_params import *
from simple_funcs import *

from networks.dataset.dataset_generator import generate_simple_dataset_from_function_object
from networks.dataset.dataset_files import load_dataset

from networks.models.simple_dense import SimpleDenseModel

from networks.visualizers.simple_dense_visualizer import simple_dense_visualizer

def train_loop(input_dict):
	model_params = input_dict["model_params"]
	model_params["optimizer"] = Adam(**model_params["optimizer_args"])
	train_data = input_dict["train_data"]
	train_labels = input_dict["train_labels"]
	test_data = input_dict["test_data"]
	test_labels = input_dict["test_labels"]
	num_epochs = input_dict["num_epochs"]
	train_batch_size = min(input_dict["train_batch_size"], len(train_data)/2)
	test_batch_size = min(input_dict["test_batch_size"], len(test_data)/2)
	epochs_per_test = input_dict["epochs_per_test"]
	num_test_vis = input_dict["num_test_vis"]
	output_dir = input_dict["output_dir"]
	F = input_dict["function"]
	prefix = input_dict["prefix"]

	model = SimpleDenseModel(model_params)
	model.load_model()
	model.train(train_data, train_labels, num_epochs, train_batch_size,
			test_data, test_labels, test_batch_size, epochs_per_test,
			num_test_vis=num_test_vis, visualizer=simple_dense_visualizer,
			output_dir=output_dir, function=F, prefix=prefix, verbosity=2)

def compute_dataset(function, train_data_size=5000, test_data_size=1000,
					train_filename=None, test_filename=None, noise_var=None):

	datasets = load_dataset(train_filename=train_filename, test_filename=test_filename)

	if test_filename is None:
		datasets_test = generate_simple_dataset_from_function_object(function, test_data_size, percent_of_train=0)
		datasets.test.data = datasets_test.test.data
		datasets.test.labels = datasets_test.test.labels

	if train_filename is None:
		datasets_train = generate_simple_dataset_from_function_object(function, train_data_size, percent_of_train=100, noise_var=noise_var)
		datasets.train.data = datasets_train.train.data
		datasets.train.labels = datasets_train.train.labels

	return datasets

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--train_data_size', type=int)
	parser.add_argument('-nt', '--num_trials', type=int)
	parser.add_argument('-ts', '--test_data')
	parser.add_argument('-fp', '--fparams_file')
	parser.add_argument('-o', '--output_dir')
	parser.add_argument('-n', '--noise_var', type=float)
	args = parser.parse_args()



	# get function_params
	function_params = load_params_file(args.fparams_file)
	# F = step_funcs.StepFunction(function_params)
	F = func_composition.CompositionFunction(function_params)

	# training params
	num_epochs = 8000
	train_batch_size = 128
	test_batch_size = 128
	epochs_per_test = 50

	# get model params
	model_params = get_model_params(F, args.train_data_size)

	datasets = [compute_dataset(F, train_data_size=args.train_data_size, test_filename=args.test_data,
								noise_var=args.noise_var) for i in range(args.num_trials)]

	# create jobs
	for i in range(args.num_trials):
		input_dict = {}
		input_dict["model_params"] = model_params

		input_dict["train_data"] = datasets[i].train.data
		input_dict["train_labels"] = datasets[i].train.labels
		input_dict["test_data"] = datasets[i].test.data
		input_dict["test_labels"] = datasets[i].test.labels

		input_dict["num_epochs"] = num_epochs
		input_dict["train_batch_size"] = train_batch_size
		input_dict["test_batch_size"] = test_batch_size
		input_dict["epochs_per_test"] = epochs_per_test

		input_dict["num_test_vis"] = 1
		input_dict["output_dir"] = args.output_dir
		input_dict["function"] = F
		input_dict["prefix"] = "trial_"+str(i)

		train_loop(input_dict)
