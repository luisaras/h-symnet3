!/bin/sh

declare -a domains=("academic_advising" "crossing_traffic" "game_of_life" "navigation" "skill_teaching" "sysadmin" "tamarisk" "traffic" "wildfire" "recon" "triangle_tireworld" "elevators")
declare -a instances=($(seq 1 10))

for d in "${domains[@]}"
do
	for i in "${instances[@]}"
	do
    	./new_instance.sh "rddl/domains/${d}_mdp.rddl" "rddl/domains/${d}_inst_mdp__${i}.rddl" "${d}_inst_mdp__${i}" $i
    done
done
