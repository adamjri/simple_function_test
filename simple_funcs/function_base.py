from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import math
import sys

try:
	from simple_funcs.utils import *
except:
	from utils import *

class FunctionBase(ABC):
	def __init__(self, params):
		self.params = params
		assert_key_in_dict("input_size", self.params)
		assert_key_in_dict("bounds", self.params)
		self.cached_function = None
		self.cached_model = None
		self.verify_params()

	def __call__(self, x):
		if self.cached_function is None:
			self.cached_function = self.get_function()
		return self.cached_function(x)

	def __getitem__(self, key):
		return self.params[key]

	@abstractmethod
	def verify_params(self):
		pass

	@abstractmethod
	def get_function(self):
		pass

	@abstractmethod
	def get_optimal_model(self):
		pass

	@property
	def optimal_model(self):
		if self.cached_model is None:
			self.cached_model = self.get_optimal_model()
		return self.cached_model

	def set_model_param(self, key, value):
		if self.cached_model is None:
			self.cached_model = self.get_optimal_model()
		self.cached_model[key] = value
