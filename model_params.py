from keras.optimizers import Adam, SGD, RMSprop, Nadam

from networks.models.simple_dense import SimpleDenseModel
from networks.models.metrics import *

import math

def get_model_params(F, train_data_size):# create model
	'''
	learning rate based on the following empirically discovered values:
	train_data_size -> empirical good learning rate -> computed learning rate
	500 -> 0.0008 -> .000806
	1000 -> 0.0009 -> 0.000926
	2500 -> 0.001 -> 0.00111
	5000 -> 0.0015 -> 0.00128
	'''
	optimizer_args = {'lr': float(train_data_size)**(1.0/5.0)/(4200.0)}

	# loss = 'binary_crossentropy'
	loss = 'mean_squared_error'
	# metrics = [mean_sq_error]
	metrics = []
	activations = F.optimal_model["activations"]
	has_biases = F.optimal_model["has_biases"]
	layers = [1, 1000, 1]
	model_params = {"layers": layers,
					"activations": activations,
					"has_biases": has_biases,
					"optimizer_args": optimizer_args,
					"loss": loss,
					"metrics": metrics}
	return model_params
