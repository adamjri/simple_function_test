import math
import numpy as np
import sys
import argparse
import os

from function_params import *
from simple_funcs import *

from networks.dataset.dataset_generator import generate_simple_dataset_from_function_object
from networks.dataset.dataset_files import save_dataset

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def generate_test_data_sweep(data_dir, output_file="test_data.txt", f_params="fparams.json", test_data_size=10000):
	f_dirs = get_immediate_subdirectories(data_dir)
	for f_dir in f_dirs:
		f_params_file = os.path.join(f_dir, f_params)
		params = load_params_file(f_params_file)
		#F = step_funcs.StepFunction(params)
		F = func_composition.CompositionFunction(params)
		dataset = generate_simple_dataset_from_function_object(F, test_data_size, percent_of_train=0)
		save_dataset(dataset, test_filename=os.path.join(f_dir, output_file))

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir')
	args = parser.parse_args()

	generate_test_data_sweep(args.data_dir)
