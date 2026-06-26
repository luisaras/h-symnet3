#!/usr/bin/env bash
# =============================

set -e

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
    if [[ ! -v ${SSIPP_ROOT} ]]; then
      export SSIPP_ROOT=$PWD/ssipp
      echo "export SSIPP_ROOT=\"${SSIPP_ROOT}\"" >> ~/.bashrc
    fi
popd

command -v podman >/dev/null 2>&1 || { echo "Error: podman is not installed." >&2; exit 1; }
podman build -t symnet-env .

if [[ ! -d "$RDDLSIM_ROOT" ]]; then
    echo "=== Installing RDDLSIM... ==="
    git clone https://github.com/ssanner/rddlsim.git "$RDDLSIM_ROOT"
    pushd "$RDDLSIM_ROOT"
        ./compile
    popd
fi
echo "RDDLSIM installed."

if [[ ! -d "$PROST_ROOT" ]]; then
    echo "=== Installing PROST... ==="
    if sudo -n true 2>/dev/null; then
        sudo apt install git g++ cmake bison flex libbdd-dev
    else
        command -v g++ >/dev/null 2>&1 || { echo "Error: g++ is not installed." >&2; exit 1; }
        command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is not installed." >&2; exit 1; }
        command -v bison >/dev/null 2>&1 || { echo "Error: bison is not installed." >&2; exit 1; }
        command -v flex >/dev/null 2>&1 || { echo "Error: flex is not installed." >&2; exit 1; }
        command -v libbdd-dev >/dev/null 2>&1 || { echo "Error: libbdd-dev is not installed." >&2; exit 1; }
    fi
    if [[ ! -d "$Z3_ROOT" ]]; then
        git clone git@github.com:Z3Prover/z3.git z3_temp
        pushd z3_temp
            python3 scripts/mk_make.py --prefix="$Z3_ROOT"
            cd build
            make
            make install
        popd
        rm -rf z3_temp
    fi
    git clone https://github.com/prost-planner/prost.git "$PROST_ROOT"
    pushd "$PROST_ROOT"
        python3 build.py
    popd
fi
echo "PROST installed."

if [[ ! -d "$SSIPP_ROOT" ]]; then
    echo "=== Installing SSIPP... ==="
    git clone https://gitlab.com/qxcv/ssipp.git "$SSIPP_ROOT"
    pushd "$SSIPP_ROOT"
        python3 build.py solver_ssp
    popd
fi
echo "SSiPP installed."
