
#rddl/domains/sysadmin_mdp.rddl
#rddl/domains/sysadmin_inst_mdp__900.rddl
#sysadmin_inst_mdp__900
#900

# domain_file instance_file instance_name
echo "relative_path_domain_file relative_path_instance_file name_of_instance instance_number"
echo "./new_instance.sh rddl/domains/sysadmin_mdp.rddl rddl/domains/sysadmin_inst_mdp__900.rddl sysadmin_inst_mdp__900 900"

# Get directories
pushd ..
symnet_path=$PWD

# Copy the instance file
(
    # Wait for lock on /var/lock/.myscript.exclusivelock (fd 200) for 10 seconds
    flock -s -x -w 300 200

    # Copy to gym
    cp ${symnet_path}/$2 ${symnet_path}/gym/envs/rddl/$2

    # Create a dbn file
    temp_folder=$PWD/temp_merger_$3
    mkdir ${temp_folder}
    cat ${symnet_path}/$1 ${symnet_path}/$2 > ${temp_folder}/temp.rddl

    # Create dbn file
    pushd ${RDDLSIM_ROOT}
    ./run rddl.viz.RDDL2Graph ${temp_folder}/temp.rddl $3
    rm -r ${temp_folder}
    # Copy the dbn file
    cp tmp_rddl_graphviz.dot ${symnet_path}/rddl/dbn/$3.dot
    cp tmp_rddl_graphviz.dot ${symnet_path}/gym/envs/rddl/rddl/dbn/$3.dot

    # Create a parsed file
    cd ${symnet_path}
    echo "STARTING RDDL-PARSER"
    ./rddl/lib/rddl-parser $1 $2 .
    echo "FINISHED WITH RDDL-PARSER"
    # Copy parsed file
    cp $3 ./rddl/parsed/$3
    cp $3 ./gym/envs/rddl/rddl/parsed/$3

    # Copy env file
    # cp ./rddl/lib/clibxx.so ./rddl/lib/clibxx$4.so 
    # cp ./rddl/lib/clibxx.so ./gym/envs/rddl/rddl/lib/clibxx$4.so 
    # chmod +x ./gym/envs/rddl/rddl/lib/clibxx$4.so

    # rm ./rddlsim-master/tmp_rddl_graphviz.dot
    rm $3
    popd


) 200>./.myscript.exclusivelock
