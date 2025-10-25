#!/bin/bash
# Quick launcher for 4-GPU setup (all GPUs)
exec "$(dirname "$0")/run_server_multigpu.sh" "HuggingFaceTB/SmolLM2-135M-Instruct" 4 0.4
