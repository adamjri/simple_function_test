import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def simple_dense_visualizer(model, test_data, test_labels, test_batch_size, output_dir, **kwargs):
	Ws = model.get_weights()
	weights_file = os.path.join(output_dir, "weights.txt")
	f = open(weights_file, 'w')
	for W in Ws:
		f.write("***************************************************************\n")
		f.write(np.array_str(W, max_line_width=1000000)+"\n")
	f.close()

	# plot_function
	model_outputs = model.predict(test_data)

	inputs_by_dimension = test_data.T

	argsorts = np.array([np.argsort(input_list) for input_list in inputs_by_dimension])
	inputs_sorted = np.array([inputs_by_dimension[i][argsorts[i]] for i in range(len(argsorts))])

	model_outputs_sorted = np.array([model_outputs[argsorts[i]].T for i in range(len(argsorts))])
	true_outputs_sorted = np.array([test_labels[argsorts[i]].T for i in range(len(argsorts))])
	for i in range(len(inputs_sorted)):
		for j in range(len(model_outputs_sorted)):
			input_list = inputs_sorted[i]
			model_output_list = model_outputs_sorted[j,i]
			true_output_list = true_outputs_sorted[j,i]
			plt.clf()
			plt.title("Output "+str(j)+" given Input "+str(i))
			line1, = plt.plot(input_list, true_output_list, 'b', label='Ground Truth '+str(j))
			line2, = plt.plot(input_list, model_output_list, 'ro', label='Predicted '+str(j))
			plt.legend(handles=[line1, line2], loc=1)
			save_filename = os.path.join(output_dir, 'output_'+str(j)+'_input_'+str(i)+'.png')
			plt.savefig(save_filename)
