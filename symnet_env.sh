#!/bin/bash
# =============================
# Run something in the symnet virtual env.
# Run ./install.sh first to create symnet-env.

if [[ "${IS_WSL}" == "true" ]]; then
	ROOT=/mnt
	GPU_FLAGS="--device \"nvidia.com/gpu=all\""
else
	ROOT=$HOME
	GPU_FLAGS="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm -v /usr/lib/x86_64-linux-gnu/nvidia:/host-nvidia:ro --env LD_LIBRARY_PATH=/host-nvidia"
	RUN_FLAG="--cdi-spec-dir=$HOME/.config/cdi"
fi
run_symnet_env() {
	podman ${RUN_FLAG} run --rm \
		--env-host \
		-v $ROOT:$ROOT \
		-w $(pwd) \
		${GPU_FLAGS} \
		--userns=keep-id \
		symnet-env \
		"$@"
}
run_symnet_env "$@"