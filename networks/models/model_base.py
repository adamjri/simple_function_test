from abc import ABC, abstractmethod, abstractproperty
from keras.callbacks import Callback
from keras.utils import plot_model

import numpy as np
import json

import datetime
import os
import sys

from networks.dataset.dataset_files import save_dataset

class TestResultsCallback(Callback):
	def __init__(self, model, test_data, test_labels, test_batch_size, epochs_per_test,
	 			results_names, train_dir=None, log_file=None,
				num_test_vis=None, visualizer=None, **kwargs):
		self.model = model
		self.test_data = test_data
		self.test_labels = test_labels
		self.test_batch_size = test_batch_size
		self.epochs_per_test = epochs_per_test
		self.results_names = results_names
		self.train_dir = train_dir
		self.log_file = log_file
		self.num_test_vis = num_test_vis
		self.visualizer = visualizer
		self.visualizer_inputs = kwargs

	def on_train_begin(self, logs={}):
		self.num_tests = 0
		self.test_results = []
		self.l_zeros = []
		if not self.log_file is None:
			header = "epoch,"
			for name in self.results_names:
				header+=name+","
			header += "l_zeros"
			f = open(self.log_file, 'w')
			f.write(header+"\n")
			f.close()

	def on_epoch_end(self, epoch, logs={}):
		if (epoch+1)%self.epochs_per_test==0:

			results = self.model.evaluate(self.test_data, self.test_labels,
										batch_size=self.test_batch_size,
										verbose=0)
			if type(results) != list:
				results = [results]

			results.insert(0, epoch)
			self.test_results.append(results)

			weights = self.model.get_weights()
			l_zero = []
			for i in range(len(weights)):
				l_zero.append(np.absolute(weights[i]).sum())
			l_zero = np.array(l_zero)
			self.l_zeros.append(l_zero)

			if not self.log_file is None:
				results_str = ""
				for i in range(len(results)):
					results_str+=str(results[i])+","
				results_str += np.array2string(l_zero, max_line_width=1000000)
				f = open(self.log_file, 'a')
				f.write(results_str+"\n")
				f.close()

			if not self.train_dir is None:
				epoch_dir = os.path.join(self.train_dir, "Epoch_"+str(epoch))
				os.makedirs(epoch_dir)
				model_file = os.path.join(epoch_dir, "model.h5")
				self.model.save(model_file)

				if not self.num_test_vis is None:
					if (self.num_tests+1)%self.num_test_vis==0:
						if not self.visualizer is None:
							self.visualizer(self.model, self.test_data, self.test_labels,
											self.test_batch_size, epoch_dir, **self.visualizer_inputs)

			self.num_tests += 1

class TrainLossCallback(Callback):
	def __init__(self, log_file=None):
		self.log_file = log_file

	def on_train_begin(self, logs={}):
		self.losses = []
		if not self.log_file is None:
			header = "loss"
			f = open(self.log_file, 'w')
			f.write(header+"\n")
			f.close()

	def on_batch_end(self, batch, logs={}):
		l = logs.get('loss')
		self.losses.append(l)
		if not self.log_file is None:
			# clear file
			f = open(self.log_file, 'a')
			f.write(str(l)+"\n")
			f.close()

class ModelBase(ABC):
	def __init__(self, params_dict):
		self.params = params_dict
		self.verify_params
		self.model = None

	@abstractmethod
	def verify_params(self):
		pass

	@abstractmethod
	def load_model(self):
		pass

	def train(self, data, labels, epochs, batch_size,
			test_data, test_labels, test_batch_size, epochs_per_test,
			num_test_vis=None, visualizer=None,
			output_dir=None, **kwargs):

		np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

		train_params = {}
		train_params["sample_size"] = len(data)+len(test_data)
		train_params["train_sample_size"] = len(data)
		train_params["test_sample_size"] = len(test_data)
		train_params["num_epochs"] = epochs
		train_params["batch_size"] = batch_size
		train_params["num_epochs_per_test"] = epochs_per_test
		train_params["test_batch_size"] = test_batch_size

		serializable_model_params = {}
		for p in self.params:
			try:
				p_str = json.dumps(self.params[p])
			except:
				p_str = self.params[p].__class__.__name__
			finally:
				serializable_model_params[p] = p_str


		# create log files
		timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
		if "prefix" in kwargs:
			timestamp = kwargs["prefix"]+"_"+timestamp
		train_dir = os.path.join(output_dir, timestamp)
		train_log_file = None
		test_log_file = None
		model_params_file = None
		train_params_file = None
		train_data_file = None
		test_data_file = None
		if not output_dir is None:
			os.makedirs(train_dir)
			train_log_file = os.path.join(train_dir, "train_log.txt")
			test_log_file = os.path.join(train_dir, "test_log.txt")

			model_params_file = os.path.join(train_dir, "model_params.txt")
			f = open(model_params_file, 'w')
			f.write(json.dumps(serializable_model_params))
			f.close()

			train_params_file = os.path.join(train_dir, "train_params.txt")
			f = open(train_params_file, 'w')
			f.write(json.dumps(train_params))
			f.close()

			train_data_file = os.path.join(train_dir, "train_data.txt")
			test_data_file = os.path.join(train_dir, "test_data.txt")
			C = type('type_C', (object,), {})
			dataset = C()
			dataset.train = C()
			dataset.test = C()
			dataset.train.data = data
			dataset.train.labels = labels
			dataset.test.data = test_data
			dataset.test.labels = test_labels
			save_dataset(dataset, train_filename=train_data_file, test_filename=test_data_file)

			model_file = os.path.join(train_dir, "model.png")
			plot_model(self.model, to_file=model_file, show_shapes=True)
			model_json_file = os.path.join(train_dir, "model.json")
			model_str = self.model.to_json()
			f = open(model_json_file, 'w')
			f.write(model_str)
			f.close()

			# weights from optimal_model
			if 'function' in kwargs:
				func_weights = []
				for i in range(len(kwargs["function"].optimal_model["weights"])):
					func_weights.append(kwargs["function"].optimal_model["weights"][i])
					func_weights.append(kwargs["function"].optimal_model["biases"][i])

				func_weights_file = os.path.join(train_dir, "func_weights.txt")
				f = open(func_weights_file, "w")
				for W in func_weights:
					f.write("***************************************************************\n")
					f.write(np.array_str(W.T, max_line_width=1000000)+"\n")
				f.close()


		results_names = self.model.metrics_names

		train_callback = TrainLossCallback(log_file=train_log_file)
		test_callback = TestResultsCallback(self.model, test_data, test_labels, test_batch_size,
											epochs_per_test, results_names, train_dir=train_dir,
											log_file=test_log_file, num_test_vis=num_test_vis,
											visualizer=visualizer, **kwargs)

		verbosity=2
		if "verbosity" in kwargs:
			verbosity = kwargs["verbosity"]
		self.model.fit(data, labels, verbose=verbosity, epochs=epochs,
						callbacks=[train_callback, test_callback])
