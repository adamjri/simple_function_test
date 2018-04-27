import sys

try:
	from simple_funcs.utils import *
except:
	from utils import *

try:
	from simple_funcs.function_base import FunctionBase
except:
	from function_base import FunctionBase

# ********************************************************************

class StepFunction(FunctionBase):
	def verify_params(self):
		assert_key_in_dict("step_locations", self.params)
		assert_key_in_dict("values", self.params)
		assert_key_in_dict("bounds", self.params)
		step_locations = self["step_locations"]
		values = self["values"]
		assert_len_diff(step_locations, values, 0)
		if not self["input_size"] == 1:
			sys.exit()

	def get_optimal_model(self):
		num_v = len(self["step_locations"])
		num_out = len(force_np_arr(self["values"][0]))
		layers = [1, num_v, 1]
		activations = ["sigmoid", "linear"]
		has_biases = [True, False]

		R = 100000.0
		weights_0 = np.zeros((num_v, 1))
		biases_0 = np.zeros(num_v)
		for j in range(num_v):
			weights_0[j, 0] = R
			biases_0[j] = -R*self["step_locations"][j]

		weights_1 = np.zeros((1, num_v))
		biases_1 = np.array([])
		for j in range(num_v):
			weights_1[0, j] = self["values"][j]

		weights = [weights_0, weights_1]
		biases = [biases_0, biases_1]
		model_dict = {}
		model_dict["layers"] = layers
		model_dict["weights"] = weights
		model_dict["biases"] = biases
		model_dict["activations"] = activations
		model_dict["has_biases"] = has_biases
		return model_dict

	def get_function(self):
		step_locations = self["step_locations"]
		values = self["values"]
		bounds = self["bounds"]
		def F(x):
			if not is_in_volume(x, bounds):
				return np.array([np.nan])
			output = 0.0
			for i in range(len(step_locations)):
				if x[0]>step_locations[i]:
					output+=values[i]
			return [output]

		return F
