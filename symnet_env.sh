#!/bin/bash
# =============================
# Run something in the symnet virtual env.
# Run ./install.sh first to create symnet-env.

if [[ "${IS_WSL}" == "true" ]]; then
	ROOT=/mnt
else
	ROOT=$HOME
fi
run_symnet_env() {
	podman run --rm \
		--env-host \
		-v $ROOT:$ROOT \
		-w $(pwd) \
		--device "nvidia.com/gpu=all" \
		--userns=keep-id \
		symnet-env \
		"$@"
}
run_symnet_env "$@"