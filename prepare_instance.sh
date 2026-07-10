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

    temp_folder=${PWD}/temp_${instance}
    mkdir ${temp_folder}

    # Create dbn file
    if [ ! -d "${domain_folder}/dbn" ]; then
        mkdir ${domain_folder}/dbn
    fi
    if [ ! -f "${instance_dbn}" ]; then
        echo "Generating .dot file for ${instance}..."
        cat ${domain_rddl} ${instance_rddl} > ${temp_folder}/temp.rddl
        pushd ${RDDLSIM_ROOT}
            ./run rddl.viz.RDDL2Graph "${temp_folder}/temp.rddl" ${instance}
            if [ -f "tmp_rddl_graphviz.dot" ]; then
                mv tmp_rddl_graphviz.dot ${instance_dbn}
                echo "${instance}.dot generated."
            else
                echo "ERROR generating DBN ${instance}.dot!"
            fi            
        popd
    fi

    # Create the parsed file
    if [ ! -d "${domain_folder}/parsed" ]; then
        mkdir ${domain_folder}/parsed
    fi
    if [ ! -f "${instance_parsed}" ]; then
        echo "Starting rddl-parser for ${instance}..."
        ./utils/rddl-parser ${domain_rddl} ${instance_rddl} ${temp_folder}
        if [ -f "${temp_folder}/${instance}" ]; then
            mv ${temp_folder}/${instance} ${instance_parsed}
            echo "${instance} parsed."
        else
            echo "ERROR parsing ${instance}!"
        fi
    fi

    rm -r ${temp_folder}


) 200>./.myscript.exclusivelock
