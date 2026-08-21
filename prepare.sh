#!/usr/bin/env bash
# =============================
# Prepare domain files (rddl, ppddl, dot, parsed) and precompute datasets and heuristics.
# Usage examples:
#	./symnet_env.sh ./prepare.sh
#	./symnet_env.sh ./prepare.sh generate 1 254
#	./symnet_env.sh ./prepare.sh parse 1 254
#	./symnet_env.sh ./prepare.sh plan 1 210
#	./symnet_env.sh ./prepare.sh plan 251 254

set -e

domains_ipc="academic_advising crossing_traffic game_of_life navigation skill_teaching sysadmin tamarisk traffic wildfire"
domains_extra="recon triangle_tireworld elevators"
domains_lr="academic_advising_chain academic_advising_prob pizza_delivery pizza_delivery_grid pizza_delivery_windy wall stochastic_navigation stochastic_wall corridor"

#declare -a domains=(${domains_ipc})
declare -a domains=("navigation")

prepare() {
	case "$1" in
		"generate")
			generate_instances
			;;
		"parse")
			set_instances $2 $3
			preprocess_rddl
			;;
		"plan")
			set_instances $2 $3
			generate_traces replan
			;;
		"heuristics")
			set_instances $2 $3
			for d in "${domains[@]}"; do
				compute_heuristics $d
			done
			;;
		*)
			set_instances $1 $2
			generate_instances
			preprocess_rddl skip
			generate_traces
			;;
	esac
}

set_instances() {
	if [[ -z "$1" ]]; then
		first_inst=1
	else
		first_inst=$1
	fi
	if [[ -z "$2" ]]; then
		last_inst=254
	else
		last_inst=$2
	fi
	num_inst=$((last_inst - first_inst + 1))
	declare -ga instances=($(seq ${first_inst} ${last_inst}))
	echo "Preparing instances ${first_inst} to ${last_inst} of domains: ${domains[@]}"
}

generate_instances() {
	echo "Generating RDDL instances."
	pushd generators
		for d in "${domains[@]}"; do
			python3 generate.py ${d} 1 -n 200 -d "train" -s "-1" -v
			python3 generate.py ${d} 201 -n 10 -d "val" -s "-1" -v
			python3 generate.py ${d} 211 -n 40 -d "test" -s "-1" -v
			python3 generate.py ${d} 251 -d "debug1" -s "-1" -v
			python3 generate.py ${d} 252 -d "debug2" -s "-1" -v
			python3 generate.py ${d} 253 -d "debug3" -s "-1" -v
			python3 generate.py ${d} 254 -d "debug4" -s "-1" -v
		done
	popd
}

preprocess_rddl() {
	echo "Generate DBN files."
	for d in "${domains[@]}"; do
		for i in "${instances[@]}"
		do
			./parse_instance.sh ${d} $i $1
		done
	done
}

plan() {
	d=$1 # domain
	if [ ! -d "data/logs/$d" ]; then
		mkdir data/logs/$d
	fi
	echo "Running prost for $d..."
	printf '%s\n' "${instances[@]}" | xargs -I {} -P 8 \
		python3 prost/run_prost.py $d {} -n 1 -p {} -d benchmarks/$d/rddl -l data/logs/$d
	if [ ! -d "data/datasets/$d" ]; then
		mkdir data/datasets/$d
	fi
	python3 prost/dataset_builder.py $d ${first_inst} -n ${num_inst} -d "data/datasets/$d" -l data/logs/$d
	compute_heuristics $d
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
			plan $d
		elif [ "$1" == "replan" ]; then
			plan $d
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
		python3 heuristics/compute_heuristics.py $d {}
}

prepare $1 $2 $3