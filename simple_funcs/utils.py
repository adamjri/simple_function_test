import numpy as np
import random as rand
import math
import sys

def assert_key_in_dict(key, dict_):
	if not key in dict_:
		print("key "+key+" required")
		sys.exit()

def force_np_arr(val):
	if not type(val).__module__=="numpy":
		if type(val) == list:
			return np.array(val)
		else:
			return np.array([val])
	return val

def assert_1d_arr(np_arr):
	if len(np_arr.shape)>1:
		print("Array must be 1D: "+str(np_arr))
		sys.exit()

def assert_len_diff(np_arr_1, np_arr_2, diff):
	if not len(np_arr_2)-len(np_arr_1)==diff:
		print("Length of array 2 - length of array 1 must = "+str(diff))
		sys.exit()

def assert_non_zero_volume(volume):
	x_1 = force_np_arr(volume[0])
	x_2 = force_np_arr(volume[1])
	assert_1d_arr(x_1)
	assert_1d_arr(x_2)
	assert_len_diff(x_1, x_2, 0)
	for i in range(len(x_1)):
		if x_1[i]==x_2[i]:
			print("Volume must be non-zero. Endpoints: "+str(x_1)+" and "+str(x_2))
			sys.exit()

def is_in_volume(point, volume):
	assert_non_zero_volume(volume)
	x_1 = force_np_arr(volume[0])
	x_2 = force_np_arr(volume[1])
	p = force_np_arr(point)
	assert_1d_arr(p)
	assert_len_diff(p, x_2, 0)
	for i in range(len(p)):
		min_ = min(x_1[i], x_2[i])
		max_ = max(x_1[i], x_2[i])
		if p[i]<min_:
			return False
		elif p[i]>=max_:
			return False
	return True

def assert_non_overlapping_volume(volume_1, volume_2):
	EPSILON = 0.00000000001
	assert_non_zero_volume(volume_1)
	x_1 = force_np_arr(volume_1[0])
	x_2 = force_np_arr(volume_1[1])
	min_vec = np.array([min(x_1[i],x_2[i]) for i in range(len(x_1))])
	max_vec = np.array([max(x_1[i],x_2[i])-EPSILON for i in range(len(x_1))])
	if is_in_volume(min_vec, volume_2):
		print("Volumes overlap: "+str(volume_1)+", "+str(volume_2))
		sys.exit()
	elif is_in_volume(max_vec, volume_2):
		print("Volumes overlap: "+str(volume_1)+", "+str(volume_2))
		sys.exit()

def assert_disjoint_volumes(volumes):
	for i in range(len(volumes)-1):
		for j in range(i+1, len(volumes)):
			assert_non_overlapping_volume(volumes[i], volumes[j])

def assert_same_lens(arrs):
	prev = force_np_arr(arrs[0])
	assert_1d_arr(prev)
	for i in range(1,len(arrs)):
		current = force_np_arr(arrs[i])
		assert_len_diff(prev, current, 0)
		prev = current
		assert_1d_arr(prev)
