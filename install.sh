#!/usr/bin/env bash
# =============================

set -e

# If an installation path is not defined, use default.
pushd ..
    if [[ ! -v ${RDDLSIM_ROOT} ]]; then
      export RDDLSIM_ROOT=$PWD/rddlsim
      echo "export RDDLSIM_ROOT=\"${RDDLSIM_ROOT}\"" >> ~/.bashrc
    fi
    if [[ ! -v ${PROST_ROOT} ]]; then
      export PROST_ROOT=$PWD/prost
      echo "export PROST_ROOT=\"${PROST_ROOT}\"" >> ~/.bashrc
    fi
    if [[ ! -v ${Z3_ROOT} ]]; then
      export Z3_ROOT=$PWD/z3-prover
      echo "export Z3_ROOT=\"${Z3_ROOT}\"" >> ~/.bashrc
    fi
popd

if [[ "${IS_WSL}" == "true" ]]; then
    ROOT=/mnt
else
    ROOT=$HOME
fi
run_env_bash() {
    podman run --rm \
        --env-host \
        -v $ROOT:$ROOT \
        -w $(pwd) \
        --userns=keep-id \
        symnet-env \
        bash -c "$@"
}

install_rddlsim() {
    if [[ ! -d "${RDDLSIM_ROOT}" ]]; then
        echo "=== Installing RDDLSIM... ==="
        git clone https://github.com/ssanner/rddlsim.git "${RDDLSIM_ROOT}"
        pushd "${RDDLSIM_ROOT}"
            run_env_bash "./compile"
        popd
    fi
    echo "RDDLSIM installed."
}

install_z3() {
    git clone https://github.com/Z3Prover/z3.git z3_temp
    pushd z3_temp
        run_env_bash "
        git checkout z3-4.8.17
        python3 scripts/mk_make.py --prefix="${Z3_ROOT}"
        cd build
        make
        make install
        "
    popd
    rm -rf z3_temp
}

install_prost() {
    if [[ ! -d "${PROST_ROOT}" ]]; then
        echo "=== Installing PROST... ==="
        if [[ ! -d "${Z3_ROOT}" ]]; then
            install_z3
        fi
        git clone https://github.com/prost-planner/prost.git "${PROST_ROOT}"
        pushd "${PROST_ROOT}"
            run_env_bash "python3 build.py"
        popd
    fi
    echo "PROST installed."
}

command -v podman >/dev/null 2>&1 || { echo "Error: podman is not installed." >&2; exit 1; }
podman build --ulimit nofile=65536:65536 -t symnet-env .
echo "Podman build complete."
install_rddlsim
install_prost

# WSL patch
if [[ "${IS_WSL}" == "true" ]]; then
    mkdir -p /run/user/$(id -u)
    chown $(id -u):$(id -g) /run/user/$(id -u)
    sudo loginctl enable-linger $(id -un)
else
    
fi
