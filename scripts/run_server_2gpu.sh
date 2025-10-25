#!/bin/bash
# Quick launcher for 2-GPU setup
exec "$(dirname "$0")/run_server_multigpu.sh" "HuggingFaceTB/SmolLM2-135M-Instruct" 2 0.4
