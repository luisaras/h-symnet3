#!/usr/bin/env bash
# =============================

set -e

ENVNAME=~/.symnet_env
if [[ ! -v ${RDDLSIM_ROOT} ]]; then
  export RDDLSIM_ROOT=../rddlsim
fi
if [[ ! -v ${RDDLSIM_ROOT} ]]; then
  export PROST_ROOT=../prost
fi
if [[ ! -v ${Z3_ROOT} ]]; then
  export Z3_ROOT=../z3-master
fi

if ! command -v python3.10 &>/dev/null; then
    # Python3.10 not installed
    if ! command -v uv &>/dev/null; then
        echo "=== Installing uv ==="
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    echo "uv installed."

    echo "=== Installing Python3.10 ==="
    uv python install 3.10
fi
echo "Python3.10 installed."

if [[ ! -d "$ENVNAME" ]]; then
    uv venv $ENVNAME --python 3.10
    echo "Virtual environment $ENVNAME created."
    nvidia_lib=$PWD/$ENVNAME/lib/python3.10/site-packages/nvidia
    echo "export LD_LIBRARY_PATH=${nvidia_lib}/cuda_runtime/lib:\${LD_LIBRARY_PATH}" >> $ENVNAME/bin/activate
    echo "export LD_LIBRARY_PATH=${nvidia_lib}/cublas/lib:\${LD_LIBRARY_PATH}" >> $ENVNAME/bin/activate
    echo "export LD_LIBRARY_PATH=${nvidia_lib}/cudnn/lib:\${LD_LIBRARY_PATH}" >> $ENVNAME/bin/activate
fi
source $ENVNAME/bin/activate
echo "Virtual environment $ENVNAME activated."

echo "=== Installing Requirements... ==="
uv pip install -r requirements.txt
uv pip install nvidia-cuda-runtime-cu11
uv pip install nvidia-cudnn-cu11
uv pip install nvidia-cublas-cu11

if [[ ! -d "$RDDLSIM_ROOT" ]]; then
    echo "=== Installing RDDLSIM... ==="
    git clone https://github.com/ssanner/rddlsim.git "$RDDLSIM_ROOT"
    pushd "$RDDLSIM_ROOT"
        ./compile
        export RDDLSIM_ROOT=$PWD
    popd
    echo "export RDDLSIM_ROOT=${RDDLSIM_ROOT}" >> ~/.bashrc
fi
echo "RDDLSIM installed."

if [[ ! -d "$PROST_ROOT" ]]; then
    echo "=== Installing PROST... ==="
    #sudo apt install git g++ cmake bison flex libbdd-dev
    command -v g++ >/dev/null 2>&1 || { echo "Error: g++ is not installed." >&2; exit 1; }
    command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is not installed." >&2; exit 1; }
    command -v bison >/dev/null 2>&1 || { echo "Error: bison is not installed." >&2; exit 1; }
    command -v flex >/dev/null 2>&1 || { echo "Error: flex is not installed." >&2; exit 1; }
    command -v libbdd-dev >/dev/null 2>&1 || { echo "Error: libbdd-dev is not installed." >&2; exit 1; }
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
        export PROST_ROOT=$PWD
    popd
    echo "export PROST_ROOT=${PROST_ROOT}" >> ~/.bashrc
fi
echo "PROST installed."