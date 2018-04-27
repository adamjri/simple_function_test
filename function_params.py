from simple_funcs import *

from random import gauss
import math
import os
import sys
import json

import numpy as np

def get_func_composition_params():
	# create function
	f_params = {}
	f_params["complexity"] = 2
	f_params["bounds"] = (-5, 5)
	f_params["input_size"] = 1

	f_params["func_layers"] = [1, 1, 1]
	f_params["has_biases"] = [True, False]
	f_params["weight_mean"] = 0.0
	f_params["weight_var"] = 1.5
	f_params["bias_mean"] = 0.0
	f_params["bias_var"] = 0.0
	f_params["activations"] = ["sigmoid", "linear"]
	return f_params

# get function parameters in a complexity sweep
def get_function_params_sweep(max_d, min_d=1, step_size=1):
	f_params_list = []
	for i in range(min_d, max_d, step_size):
		f_params = {}
		f_params["complexity"] = i
		f_params["bounds"] = (-1, 1)
		f_params["input_size"] = 1
		f_params["step_locations"] = [2.0*(j+1.0)/(i+1.0) - 1.0 for j in range(i)]
		f_params["values"] = [gauss(0.0, 6.0) for j in range(i)]
		f_params_list.append(f_params)
	return f_params_list

# get function parameters in a complexity sweep
def get_func_composition_params_sweep(max_d, min_d=1, step_size=1):
	f_params_list = []
	for i in range(min_d, max_d, step_size):
		f_params = {}
		f_params["complexity"] = i
		f_params["bounds"] = (-1, 1)
		f_params["input_size"] = 1

		f_params["func_layers"] = [1, i, 1]
		f_params["has_biases"] = [True, False]
		f_params["weight_mean"] = 0.0
		f_params["weight_var"] = 10.0
		f_params["bias_mean"] = 0.0
		f_params["bias_var"] = 10.0
		f_params["activations"] = ["sigmoid", "linear"]
		F = func_composition.CompositionFunction(f_params)
		weights = [w.tolist() for w in F.optimal_model["weights"]]
		biases = [b.tolist() for b in F.optimal_model["biases"]]
		f_params["weights"] = weights
		f_params["biases"] = biases
		f_params_list.append(f_params)
	return f_params_list

def save_params_file(filename, params):
	f = open(filename, 'w')
	json.dump(params, f)
	f.close()

def load_params_file(filename):
	f = open(filename, 'r')
	params = json.load(f)
	f.close()
	return params

def save_params_list(dump_dir, params_list):
	for params in params_list:
		print(params["complexity"])
		f_dir = os.path.join(dump_dir, "d"+str(params["complexity"]) )
		if not os.path.exists(f_dir):
			os.makedirs(f_dir)
		fname = os.path.join(f_dir, "fparams.json")
		save_params_file(fname, params)

if __name__ == "__main__":
	dump_dir = sys.argv[1]

	if len(sys.argv)==3:
		min_d = 0
		max_d = int(sys.argv[2])
		step_size = 1
	elif len(sys.argv)==4:
		min_d = int(sys.argv[2])
		max_d = int(sys.argv[3])
		step_size = 1
	elif len(sys.argv)>4:
		min_d = int(sys.argv[2])
		max_d = int(sys.argv[3])
		step_size = int(sys.argv[4])

	# save_params_list(dump_dir, get_function_params_sweep(max_d, min_d=min_d, step_size=step_size))
	save_params_list(dump_dir, get_func_composition_params_sweep(max_d, min_d=min_d, step_size=step_size))
