#!/usr/bin/env bash
# =============================

set -e

domains_ipc="academic_advising crossing_traffic game_of_life navigation skill_teaching sysadmin tamarisk traffic wildfire"
domains_extra="recon triangle_tireworld elevators"
domains_lr="academic_advising_chain academic_advising_prob pizza_delivery pizza_delivery_grid pizza_delivery_windy wall stochastic_navigation stochastic_wall corridor"

#declare -a domains=(${domains_ipc})
num_instances=250
declare -a domains=("navigation")
declare -a instances=($(seq 1 ${num_instances}))
#declare -a instances
#for ((i=1; i<=num_instances; i++)); do
#	instances+=("$i")
#done

generate_rddl() {
	echo "Generating RDDL instances."
	pushd generators
		for d in "${domains[@]}"
		do
			python3 generate.py ${d} 1 -n 200 -d "train" -s "-1" --skip -v
			python3 generate.py ${d} 201 -n 10 -d "val" -s "-1" --skip -v
			python3 generate.py ${d} 211 -n 40 -d "test" -s "-1" --skip -v
		done
	popd
}

preprocess_rddl() {
	echo "Generate DBN and PPDDL files."
	for d in "${domains[@]}"
	do
		for i in "${instances[@]}"
		do
			./prepare_instance.sh ${d} $i
		done
	done
}

generate_traces() {
	echo "Create trace datasets with PROST."
	if [ ! -d "data" ]; then
		mkdir data
		mkdir data/datasets
		mkdir data/logs
	fi
	for d in "${domains[@]}"
	do
		if [ ! -d "data/datasets/${d}" ]; then
			mkdir data/datasets/${d}
			mkdir data/logs/${d}
			echo "Running prost for ${d}..."
			parallel -j 8 python3 prost/run_prost.py ${d} {} -n 1 -p {} -d benchmarks/${d}/rddl -l data/logs/${d} ::: ${instances[@]}
			#for i in "${instances[@]}"; do
			#	python3 prost/run_prost.py ${d} 1 -n ${num_instances} -d benchmarks/${d}/rddl -l data/logs/${d}
			#done
			#python $PROST_ROOT ${d}_inst_mdp__${i} [Prost -s 1 -se [IPC2014]] >> data/logs/${d}/${i}.result
			python3 prost/dataset_builder.py --domain ${d} --save_folder "data/datasets/${d}" --prost_log data/logs/${d} --start_instance 1 --num_instances ${num_instances}
		fi
	done
}

case "$1" in
    "generate")
        generate_rddl
        ;;
    "parse")
        preprocess_rddl
        ;;
    "plan")
    	rm -rf data
        generate_traces
        ;;
    *)
    	generate_traces
    	preprocess_rddl
        generate_traces
        ;;
esac
