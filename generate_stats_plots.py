import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import math

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_avg(value_list):
	d = 0.0
	sum_ = 0.0
	for v in value_list:
		sum_+=v
		d+=1.0
	return sum_/d

def load_log_file(filename):
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	header = lines[0]
	header_list = header[:-1].split(",")
	output_dict = {}
	for key in header_list:
		output_dict[key] = []
	for line in lines[1:]:
		value_list = line[:-1].split(",")
		for i in range(len(value_list)):
			key = header_list[i]
			value_str = value_list[i]
			if value_str[0]=="[":
				value = np.fromstring(value_str[1:-1], sep=' ')
			else:
				value = float(value_str)
			output_dict[key].append(value)
	return output_dict

def get_min_index_value(data_dict, key='loss'):
	min_v = float('Inf')
	min_i = -1
	for i in range(len(data_dict[key])):
		if data_dict[key][i]<=min_v:
			min_v = data_dict[key][i]
			min_i = i
	return [min_i, min_v]

def get_k_min_index_value_trial(trials_dir, k=1, fname='test_log.txt', key='loss'):
	trial_dirs = get_immediate_subdirectories(trials_dir)
	best_mins = [float('Inf') for j in range(k)]
	best_i = [-1 for j in range(k)]
	best_trials = ['' for j in range(k)]
	for trial_dir in trial_dirs:
		filename = os.path.join(trial_dir, fname)
		data_dict = load_log_file(filename)
		min_i, min_v = get_min_index_value(data_dict)

		max_min_v = -1
		max_min_j = -1
		for j in range(k):
			if max_min_v<=best_mins[j]:
				max_min_v=best_mins[j]
				max_min_j=j

		if min_v<max_min_v:
			best_mins[max_min_j] = min_v
			best_i[max_min_j] = min_i
			best_trials[max_min_j] = trial_dir

	return [best_i, best_mins, best_trials]

def get_data_for_function_run(function_dir, k=1, fname='test_log.txt', key='loss'):
	trials_dirs = get_immediate_subdirectories(function_dir)
	X = []
	Y = []
	for trials_dir in trials_dirs:
		num_samples = int(trials_dir.split("_")[-1])
		X.append(num_samples)
		indexes, values, trial_dirs = get_k_min_index_value_trial(trials_dir, k=k, fname=fname, key=key)
		avg_value = get_avg(values)
		Y.append(avg_value)
	# sort X and Y by X
	np_X = np.array(X)
	np_Y = np.array(Y)
	sortlist = np.argsort(np_X)
	sorted_X = np_X[sortlist]
	sorted_Y = np_Y[sortlist]
	return [sorted_X, sorted_Y]

def get_all_data(data_dir, k=1, fname='test_log.txt', key='loss'):
	function_dirs = get_immediate_subdirectories(data_dir)
	data_dict = {}
	for function_dir in function_dirs:
		f_split = function_dir.split("/")[-1]
		if f_split[0]=="d":
			function_name = f_split[1:]
			data = get_data_for_function_run(function_dir, k=k, fname=fname, key=key)
			data_dict[function_name] = data
	return data_dict

def plot_all_data(data_dict, exponent, save_file=None):
	plt.clf()
	X_data_list = []
	Y_data_list = []
	key_list = []
	for key in data_dict:
		X_data, Y_data = data_dict[key]
		value = (float(key))**exponent
		X_data_list.append(X_data)
		Y_data_list.append(Y_data/value)
		key_list.append(int(key))
	X_data_list = np.array(X_data_list)
	Y_data_list = np.array(Y_data_list)
	key_list = np.array(key_list)

	sort_list = np.argsort(key_list)

	sorted_X_list = X_data_list[sort_list]
	sorted_Y_list = Y_data_list[sort_list]
	sorted_key_list = key_list[sort_list]

	for i in range(len(sorted_key_list)):
		plt.plot(sorted_X_list[i], sorted_Y_list[i], label="d_"+str(sorted_key_list[i]))

	plt.legend(loc='best')
	if not save_file is None:
		plt.savefig(save_file)
	#plt.show()

def plot_all_data_e_vs_dN(data_dict, num_exp, denom_exp, save_file=None):
	plt.clf()
	marker = itertools.cycle(('^', '+', 'o', '*'))
	for key in data_dict:
		X_data, Y_data = data_dict[key]
		modified_X_data = []
		modified_Y_data = []
		for i in reversed(range(len(X_data))):
			#modified_X_data.append((float(key)+1.0)/X_data[i])
			modified_X_data.append(((float(key)+1.0)**num_exp)/(X_data[i]**denom_exp))
			modified_Y_data.append(Y_data[i])
		plt.plot(modified_X_data, modified_Y_data, marker=next(marker), linestyle='', label='d_'+key)
	plt.legend(loc='best')
	plt.title('Error vs (D^'+str(num_exp)+")/(N^"+str(denom_exp)+")")
	if not save_file is None:
		plt.savefig(save_file)
	#plt.show()

def float_to_string(f, size):
	f_str = str(round(f, size))
	f_tail = f_str.split(".")[1]
	diff = size - len(f_tail)
	for i in range(diff):
		f_str+="0"
	return f_str

if __name__=="__main__":
	num_top_avg = 3
	data_dir = "/scratch/richards/network_data/sweep_11"
	data_dict = get_all_data(data_dir, k=num_top_avg)

	if not os.path.exists(os.path.join(data_dir, "plots1")):
		os.makedirs(os.path.join(data_dir, "plots1"))
	num_exp_range = [0.25, 5.0, 0.25]
	denom_exp_range = [0.25, 3.0, 0.25]
	num_exp = num_exp_range[0]
	while num_exp<=num_exp_range[1]:
		denom_exp = denom_exp_range[0]
		while denom_exp<=denom_exp_range[1]:
			save_file = data_dir+"/plots1/e_vs_D^"+float_to_string(num_exp,2)+"_N^"+float_to_string(denom_exp,2)+".png"
			plot_all_data_e_vs_dN(data_dict, num_exp, denom_exp, save_file=save_file)
			denom_exp+=denom_exp_range[2]
		num_exp+=num_exp_range[2]

	if not os.path.exists(os.path.join(data_dir, "plots2")):
		os.makedirs(os.path.join(data_dir, "plots2"))
	exp_range = [0.5, 5.0, 0.25]
	exp = exp_range[0]
	while exp<=exp_range[1]:
		save_file = data_dir+"/plots2/e_D^"+float_to_string(exp,2)+"_vs_N.png"
		plot_all_data(data_dict, exp, save_file=save_file)
		exp+=exp_range[2]

	plot_all_data(data_dict, 0.0, save_file=data_dir+"/data_plot.png")
