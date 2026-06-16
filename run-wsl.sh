#!/bin/bash
# See run.py for arguments.
podman run --rm -v /mnt:/mnt -w $(pwd) --userns=keep-id symnet-env python run.py "$@"