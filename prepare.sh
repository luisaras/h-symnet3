#!/usr/bin/env bash
# =============================

set -e

domains_ipc="academic_advising crossing_traffic game_of_life navigation skill_teaching sysadmin tamarisk traffic wildfire"
domains_extra="recon triangle_tireworld elevators"
domains_lr="academic_advising_chain academic_advising_prob pizza_delivery pizza_delivery_grid pizza_delivery_windy wall stochastic_navigation stochastic_wall corridor"

#declare -a domains=(${domains_ipc})
num_instances=250
declare -a domains=("navigation")
#declare -a instances=($(seq 1 ${num_instances}))
declare -a instances=($(seq 300 303))
#declare -a instances
#for ((i=1; i<=num_instances; i++)); do
#	instances+=("$i")
#done

generate_rddl() {
	echo "Generating RDDL instances."
	pushd generators
		for d in "${domains[@]}"; do
			python3 generate.py ${d} 1 -n 200 -d "train" -s "-1" --skip -v
			python3 generate.py ${d} 201 -n 10 -d "val" -s "-1" --skip -v
			python3 generate.py ${d} 211 -n 40 -d "test" -s "-1" --skip -v
		done
	popd
}

preprocess_rddl() {
	echo "Generate DBN and PPDDL files."
	for d in "${domains[@]}"; do
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
	for d in "${domains[@]}"; do
		if [ ! -d "data/datasets/$d" ]; then
			mkdir data/datasets/$d
			mkdir data/logs/$d
			echo "Running prost for $d..."
			printf '%s\n' "${instances[@]}" | xargs -I {} -P 8 \
				python3 prost/run_prost.py $d {} -n 1 -p {} -d benchmarks/$d/rddl -l data/logs/$d
			python3 prost/dataset_builder.py $d 1 -n ${num_instances} -d "data/datasets/$d" -l data/logs/$d
			compute_heuristics $d
		fi
	done
}

compute_heuristics() {
	echo "Compute heuristics from PROST traces."
	if [ ! -d "data/heuristics" ]; then
		mkdir data/heuristics
	fi
	d=$1 # domain
	if [ ! -d "data/heuristics/$d" ]; then
		mkdir data/heuristics/$d
	fi
	echo "Computing heuristics for $d..."
	printf '%s\n' "${instances[@]}" | xargs -I {} -P 8 \
		python3 utils/compute_heuristics.py $d {}
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
	"heuristics")
		for d in "${domains[@]}"; do
			compute_heuristics $d
		done
		;;
	*)
		generate_traces
		preprocess_rddl
		generate_traces
		;;
esac
