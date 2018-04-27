import numpy as np
import os
import sys

def save_dataset(dataset, train_filename=None, test_filename=None):
	if not train_filename is None:
		f = open(train_filename, 'w')
		f.write('data,labels\n')
		for i in range(len(dataset.train.data)):
			f.write(np.array2string(dataset.train.data[i], max_line_width=1000000)+","+
					np.array2string(dataset.train.labels[i], max_line_width=1000000)+"\n")
		f.close()
	if not test_filename is None:
		f = open(test_filename, 'w')
		f.write('data,labels\n')
		for i in range(len(dataset.test.data)):
			f.write(np.array2string(dataset.test.data[i], max_line_width=1000000)+","+
					np.array2string(dataset.test.labels[i], max_line_width=1000000)+"\n")
		f.close()

def load_dataset(train_filename=None, test_filename=None):
	C = type('type_C', (object,), {})
	dataset = C()
	dataset.train = C()
	dataset.test = C()
	dataset.train.data=None
	dataset.train.labels=None
	dataset.test.data=None
	dataset.test.labels=None
	if not train_filename is None:
		f = open(train_filename, 'r')
		lines = f.readlines()
		header = lines[0]
		fields = header[:-1].split(",")
		if not 'data' in fields:
			print("No data field in dataset file")
			sys.exit()
		if not 'labels' in fields:
			print("No labels field in dataset file")
			sys.exit()
		data_index = fields.index('data')
		labels_index = fields.index('labels')
		data_list = []
		labels_list = []
		for line in lines[1:]:
			line_list = line[:-1].split(",")
			data_list.append(np.fromstring(line_list[data_index][1:-1], sep=" "))
			labels_list.append(np.fromstring(line_list[labels_index][1:-1], sep=" "))
		dataset.train.data = np.array(data_list)
		dataset.train.labels = np.array(labels_list)

	if not test_filename is None:
		f = open(test_filename, 'r')
		lines = f.readlines()
		header = lines[0]
		fields = header[:-1].split(",")
		if not 'data' in fields:
			print("No data field in dataset file")
			sys.exit()
		if not 'labels' in fields:
			print("No labels field in dataset file")
			sys.exit()
		data_index = fields.index('data')
		labels_index = fields.index('labels')
		data_list = []
		labels_list = []
		for line in lines[1:]:
			line_list = line[:-1].split(",")
			data_list.append(np.fromstring(line_list[data_index][1:-1], sep=" "))
			labels_list.append(np.fromstring(line_list[labels_index][1:-1], sep=" "))
		dataset.test.data = np.array(data_list)
		dataset.test.labels = np.array(labels_list)

	return dataset
