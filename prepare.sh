#!/usr/bin/env bash
# =============================

set -e

declare -a lr_domains=("academic_advising_chain" "academic_advising_prob" "pizza_delivery" "pizza_delivery_grid" "pizza_delivery_windy" "wall" "stochastic_navigation" "stochastic_wall" "corridor")
declare -a domains_extra=("recon" "triangle_tireworld" "elevators")
declare -a domains=("academic_advising" "crossing_traffic" "game_of_life" "navigation" "skill_teaching" "sysadmin" "tamarisk" "traffic" "wildfire")
declare -a instances=($(seq 1 10))

for d in "${domains[@]}"
do
	for i in "${instances[@]}"
	do
    	./prepare_instance.sh ${d} $i
    done
done


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
	    python prost/dataset_builder.py --domain ${d} --save_folder "data/datasets/${d}" --prost_log data/logs/${d} --start_instance 0 --num_instances 10
	done
fi