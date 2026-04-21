#!/usr/bin/env bash
# =============================

set -e

declare -a domains=("academic_advising" "crossing_traffic" "game_of_life" "navigation" "skill_teaching" "sysadmin" "tamarisk" "traffic" "wildfire" "recon" "triangle_tireworld" "elevators")
declare -a instances=($(seq 1 10))

if [ ! -d "rddl" ]; then
	mkdir rddl
	mkdir rddl/lib
	mkdir rddl/dbn
	mkdir rddl/parsed
	cp $PROST_ROOT/builds/release/rddl_parser/rddl-parser rddl/lib
	mkdir rddl/domains
	cp $PROST_ROOT/testbed/benchmarks/*/*.rddl rddl/domains
	pushd utils
	for d in "${domains[@]}"
	do
		for i in "${instances[@]}"
		do
	    	./new_instance.sh "rddl/domains/${d}_mdp.rddl" "rddl/domains/${d}_inst_mdp__${i}.rddl" "${d}_inst_mdp__${i}" $i
	    done
	done
	popd
fi

if [ ! -d "data" ]; then
	mkdir data
	mkdir data/datasets
	mkdir data/logs
	python3.10 "$PROST_ROOT/testbed/run-server.py" -b rddl/domains -r 100 &
	for d in "${domains[@]}"
	do
		mkdir data/datasets/${d}
		mkdir data/logs/${d}
		for i in "${instances[@]}"
		do
			python $PROST_ROOT ${d}_inst_mdp__${i} [Prost -s 1 -se [IPC2014]] >> data/logs/${d}/${i}.result
	    done
	    python multi_train/deep_plan/dataset_builder.py --domain ${d} --save_folder "data/datasets/${d}" --prost_log data/logs/${d} --start_instance 0 --num_instances 10
	done
fi