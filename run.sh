#!/bin/bash
# See run.py for arguments.
echo $(pwd)
podman run --rm -v $HOME:$HOME -w $(pwd) --device "://nvidia.com" --userns=keep-id symnet-env run python run.py "$@"