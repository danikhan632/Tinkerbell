#!/bin/bash
# Start Tink server with background vLLM process
#
# Usage:
#   ./start_with_vllm.sh                    # Single GPU
#   ./start_with_vllm.sh --nproc 2          # 2 GPUs with Megatron
#

set -e

# Default configuration
NPROC=${NPROC:-1}
PORT=${PORT:-8000}
VLLM_PORT=${VLLM_PORT:-8001}
MODEL=${MODEL:-"HuggingFaceTB/SmolLM2-135M-Instruct"}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================"
echo "Starting Tink with vLLM"
echo "================================"
echo "Configuration:"
echo "  GPUs: $NPROC"
echo "  Flask port: $PORT"
echo "  vLLM port: $VLLM_PORT"
echo "  Model: $MODEL"
echo ""

# Export environment variables
export PORT=$PORT
export VLLM_AUTO_START=true
export VLLM_PORT=$VLLM_PORT
export VLLM_MODEL=$MODEL
export VLLM_HOST=0.0.0.0
export VLLM_ENABLE_LORA=true
export VLLM_MAX_LORAS=4
export VLLM_MAX_LORA_RANK=64
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_GPU_MEMORY_UTIL=0.2  # Only 5% - leaves 95% for training!
export VLLM_DTYPE=auto

# Enable vLLM backend for worker
export USE_VLLM=true
export VLLM_BASE_URL=http://localhost:$VLLM_PORT

# Megatron settings (if using multiple GPUs)
if [ $NPROC -gt 1 ]; then
    echo "Multi-GPU setup detected. Enabling Megatron..."
    export USE_MEGATRON=true
    export MEGATRON_SERVER_URL=http://localhost:5000
else
    echo "Single GPU setup. Using HuggingFace backend..."
    export USE_MEGATRON=false
fi

# Worker pool size
export MAX_WORKERS=4

echo ""
echo "Starting server..."
echo ""

# Run with torchrun
torchrun \
    --nproc_per_node=$NPROC \
    --standalone \
    src/app.py

# Cleanup is handled by atexit in vllm_process_manager.py
