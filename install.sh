#!/usr/bin/env bash
# =============================

set -e

ENVNAME=".venv"
if [[ ! -v ${RDDLSIM_ROOT} ]]; then
  export RDDLSIM_ROOT="../rddlsim"
fi
if [[ ! -v ${RDDLSIM_ROOT} ]]; then
  export PROST_ROOT="../prost"
fi
if [[ ! -v ${Z3_ROOT} ]]; then
  export Z3_ROOT="../z3-master"
fi

if ! command -v python3.12 &>/dev/null; then
    # Python3.12 not installed
    if ! command -v uv &>/dev/null; then
        echo "=== Installing uv ==="
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    echo "uv installed."

    echo "=== Installing Python3.12 ==="
    uv python install 3.12
fi
echo "Python3.10 installed."

if [[ ! -d "$ENVNAME" ]]; then
    uv venv $ENVNAME --python 3.10
    echo "Virtual environment $ENVNAME created."
fi
source $ENVNAME/bin/activate
echo "Virtual environment $ENVNAME activated."

echo "=== Installing Requirements... ==="
uv pip install -r requirements.txt

if [[ ! -d "$RDDLSIM_ROOT" ]]; then
    echo "=== Installing RDDLSIM... ==="
    git clone https://github.com/ssanner/rddlsim.git "$RDDLSIM_ROOT"
    pushd "$RDDLSIM_ROOT"
        ./compile
    popd
    echo "export RDDLSIM_ROOT=${RDDLSIM_ROOT}" >> ~/.bashrc
fi
echo "RDDLSIM installed."

if [[ ! -d "$PROST_ROOT" ]]; then
    echo "=== Installing PROST... ==="
    sudo apt install git g++ cmake bison flex libbdd-dev
    if [[ ! -d "$Z3_ROOT" ]]; then
        git clone git@github.com:Z3Prover/z3.git z3_temp
        pushd z3_temp
        python3 scripts/mk_make.py --prefix="$Z3_ROOT"
        cd build
        make
        make install
        popd
    fi
    git clone https://github.com/prost-planner/prost.git "$PROST_ROOT"
    pushd "$PROST_ROOT"
        ./build.py
    popd
    echo "export PROST_ROOT=${PROST_ROOT}" >> ~/.bashrc
fi
echo "PROST installed."