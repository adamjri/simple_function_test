import numpy as np
import sys
import math

def shuffle_in_unison_inplace(a, b):
	"""Shuffle the arrays randomly"""
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def data_shuffle(data_sets_org, percent_of_train, shuffle_data=False):
	"""Divided the data to train and test and shuffle it"""
	perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
	C = type('type_C', (object,), {})
	data_sets = C()
	stop_train_index = perc(percent_of_train, len(data_sets_org["data"]))
	start_test_index = stop_train_index
	data_sets.train = C()
	data_sets.test = C()
	if shuffle_data:
		shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org["data"], data_sets_org["labels"])
	else:
		shuffled_data, shuffled_labels = data_sets_org["data"], data_sets_org["labels"]
	data_sets.train.data = shuffled_data[:stop_train_index, :]
	data_sets.train.labels = shuffled_labels[:stop_train_index, :]
	data_sets.test.data = shuffled_data[start_test_index:, :]
	data_sets.test.labels = shuffled_labels[start_test_index:, :]
	return data_sets

def generate_simple_dataset_from_function_object(function_object, sample_size, percent_of_train=85, noise_var=None):
	input_size = function_object["input_size"]
	if input_size>1:
		min_bounds = [min(function_object["bounds"][0][i],function_object["bounds"][1][i]) for i in range(input_size)]
		max_bounds = [max(function_object["bounds"][0][i],function_object["bounds"][1][i]) for i in range(input_size)]
	else:
		min_bounds = [min(function_object["bounds"][0],function_object["bounds"][1])]
		max_bounds = [max(function_object["bounds"][0],function_object["bounds"][1])]

	data_sets_org = {}
	data_sets_org["data"] = np.array([[np.random.random()*(max_bounds[i]-min_bounds[i])+min_bounds[i]
	 									for i in range(input_size)]
	 										for j in range(sample_size)])

	data_sets_org["labels"] = np.array([function_object(x) for x in data_sets_org["data"]])
	if not noise_var is None:
		noise = np.random.normal(0.0, math.sqrt(noise_var), data_sets_org["labels"].shape)
		data_sets_org["labels"] += noise


	datasets = data_shuffle(data_sets_org, percent_of_train)

	return datasets
