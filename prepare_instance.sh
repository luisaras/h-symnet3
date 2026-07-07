#!/usr/bin/env bash
# =============================
# Usage example: ./prepare_instance.sh navigation 15 

(
    # Wait for lock on /var/lock/.myscript.exclusivelock (fd 200) for 10 seconds
    flock -s -x -w 300 200

    domain=$1
    instance=${domain}_inst_mdp__$2
    domain_folder="${PWD}/benchmarks/${domain}"
    domain_rddl="${domain_folder}/rddl/${domain}_mdp.rddl"
    instance_rddl="${domain_folder}/rddl/${instance}.rddl"
    instance_dbn="${domain_folder}/dbn/${instance}.dot"
    instance_parsed="${domain_folder}/parsed/${instance}"
    instance_ppddl="${domain_folder}/ppddl/${instance}.ppddl"

    temp_folder=${PWD}/temp_${instance}
    mkdir ${temp_folder}

    # Create dbn and ppddl file
    if [ ! -d "${domain_folder}/dbn" ]; then
        mkdir ${domain_folder}/dbn
    fi
    if [ ! -d "${domain_folder}/ppddl" ]; then
        mkdir ${domain_folder}/ppddl
    fi
    cat ${domain_rddl} ${instance_rddl} > ${temp_folder}/temp.rddl
    pushd ${RDDLSIM_ROOT}
    if [ ! -f "${instance_dbn}" ]; then
        ./run rddl.viz.RDDL2Graph "${temp_folder}/temp.rddl" ${instance}
        mv tmp_rddl_graphviz.dot ${instance_dbn}
    fi
    if [ ! -f "${instance_ppddl}" ]; then
        ./run rddl.translate.RDDL2Format "${temp_folder}/temp.rddl" ${temp_folder} ppddl
        mv ${temp_folder}/${instance}.ppddl ${instance_ppddl}
    fi
    popd

    # Create the parsed file
    if [ ! -d "${domain_folder}/parsed" ]; then
        mkdir ${domain_folder}/parsed
    fi
    if [ ! -f "${instance_parsed}" ]; then
        echo "STARTING RDDL-PARSER ${instance}"
        ./utils/rddl-parser ${domain_rddl} ${instance_rddl} ${temp_folder}
        echo "FINISHED WITH RDDL-PARSER"
        mv ${temp_folder}/${instance} ${instance_parsed}
    fi

    rm -r ${temp_folder}


) 200>./.myscript.exclusivelock
