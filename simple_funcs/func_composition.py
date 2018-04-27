import numpy as np
import random as rand
import math
import sys

try:
	from simple_funcs.utils import *
except:
	from utils import *

try:
	from simple_funcs.function_base import FunctionBase
except:
	from function_base import FunctionBase


# activation map
ACTIVATION_MAP = {
	"linear": lambda x: x,
	"sigmoid": lambda x: 1.0/(1.0+math.exp(-float(x))),
	"relu": lambda x: max(0.0, x),
	"softplus": lambda x: math.log(1.0+math.exp(float(x))),
}

class CompositionFunction(FunctionBase):

	def verify_params(self):
		assert_key_in_dict("func_layers", self.params)
		assert_key_in_dict("has_biases", self.params)
		assert_key_in_dict("weight_mean", self.params)
		assert_key_in_dict("weight_var", self.params)
		assert_key_in_dict("bias_mean", self.params)
		assert_key_in_dict("bias_var", self.params)
		assert_key_in_dict("activations", self.params)

		weight_var = self["weight_var"]
		bias_var = self["bias_var"]
		if weight_var<0:
			print("Weight variance must be non-negative")
			sys.exit()
		if bias_var<0:
			print("Bias variance must be non-negative")
			sys.exit()

		func_layers = self["func_layers"]
		has_biases = self["has_biases"]
		activations = self["activations"]
		assert_len_diff(activations, func_layers, 1)
		assert_len_diff(activations, has_biases, 0)

		if "weights" in self.params:
			model_weights = []
			for w in self["weights"]:
				model_weights.append(np.array(w))
			self.set_model_param("weights", model_weights)
		if "biases" in self.params:
			model_biases = []
			for b in self["biases"]:
				model_biases.append(np.array(b))
			self.set_model_param("biases", model_biases)

	def get_optimal_model(self):
		func_layers = self["func_layers"]
		weight_mean = self["weight_mean"]
		weight_var = self["weight_var"]
		bias_mean = self["bias_mean"]
		bias_var = self["bias_var"]
		has_biases = self["has_biases"]

		len_ = len(func_layers)

		weights = []
		biases = []
		for i in range(len_-1):
			layer_weights = np.array([[rand.gauss(weight_mean, math.sqrt(weight_var))
			 							for k in range(func_layers[i])]
									for j in range(func_layers[i+1])])
			if has_biases[i]:
				layer_bias = np.array([rand.gauss(bias_mean, math.sqrt(bias_var))
				 						for j in range(func_layers[i+1])])
			else:
				layer_bias = np.array([0.0 for j in range(func_layers[i+1])])
			weights.append(np.array(layer_weights))
			biases.append(np.array(layer_bias))

		model_dict = {}
		model_dict["layers"] = func_layers
		model_dict["weights"] = weights
		model_dict["biases"] = biases
		model_dict["has_biases"] = has_biases
		model_dict["activations"] = self["activations"]
		return model_dict

	def get_function(self):
		weights = self.optimal_model["weights"]
		biases = self.optimal_model["biases"]
		activations = self["activations"]
		l_ = len(weights)
		vcs = [np.vectorize(ACTIVATION_MAP[activation]) for activation in activations]
		# assume input is numpy 1D array
		def func(input_vec):
			vec = input_vec[:, np.newaxis]
			for i in range(l_):
				vec = weights[i].dot(vec) + biases[i][:, np.newaxis]
				vec = vcs[i](vec)
			return vec[:,0]
		return func
