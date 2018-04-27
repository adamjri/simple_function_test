#!/bin/bash

# delete all qsubs command
# qstat -u $(whoami) -n1 | grep ".iscsrv" | awk '{print $1;}' | while read a ; do qdel ${a%%.*} ; done

data_dir="/scratch/richards/network_data/sweep_12/"
tmp_dir="/var/tmp/"

function_params_name="fparams.json"
test_data_name="test_data.txt"

num_train_sweep="50 100 200 300 400 600"

for data_subdir in ${data_dir}*/; do
	f_params_file="${data_subdir}${function_params_name}"
	test_data_file="${data_subdir}${test_data_name}"

	for num_train in $num_train_sweep; do
		num_trials="10"

		script_output_dir="${tmp_dir}$(basename ${data_subdir})_N${num_train}/"

		# qsub command
		qsub_cmd="qsub -M richards@iscsrv15.epfl.ch -m e \
		-N $(basename ${data_subdir})_N${num_train} \
		-q long \
		-l nodes=1 \
		-l pmem=3gb \
		-l cput=4:00:00 \
		-o ${data_subdir}stdout_N${num_train}.out \
		-e ${data_subdir}stderr_N${num_train}.out"

		# change directory to python script directory
		cd_cmd="cd simple_function_test"
		# activate virtual environment
		env_cmd="source activate keras_py3_6"
		# clear out output dir
		clr_cmd="rm -rf ${script_output_dir}"
		# create output_dir on local node
		mkdir_cmd="mkdir ${script_output_dir}"
		# Python script command
		python_cmd="python run_from_qsub.py -tr ${num_train} \
		-nt ${num_trials} -ts ${test_data_file} \
		-fp ${f_params_file} -o ${script_output_dir}"


		# copy outputs to scratch space
		mv_cmd="mv ${script_output_dir} ${data_subdir}train_${num_train}"

		input_cmd_str="${cd_cmd}; ${env_cmd}; ${clr_cmd}; ${mkdir_cmd}; ${python_cmd}; ${mv_cmd}"

		full_cmd="echo \"${input_cmd_str}\" | ${qsub_cmd}"
		echo "*****************************************************************"
		echo $full_cmd
		eval $full_cmd

		sleep 0.2
	done

done
