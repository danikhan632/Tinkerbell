#!/bin/bash
#
# Multi-GPU Tinkerbell Server Startup Script
#
# This script starts the Tinkerbell server with vLLM configured for multi-GPU tensor parallelism.
# vLLM will automatically distribute the model across GPUs for faster inference.
#
# Usage:
#   ./scripts/run_server_multigpu.sh [model_name] [num_gpus] [gpu_memory_util]
#
# Examples:
#   ./scripts/run_server_multigpu.sh                                    # Use defaults
#   ./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 2         # 2 GPUs
#   ./scripts/run_server_multigpu.sh Qwen/Qwen3-0.6B 4 0.8             # 4 GPUs, 80% memory
#

set -e  # Exit on error
export CUDA_HOME=/usr/local/cuda-12
export CUDA_PATH=/usr/local/cuda-12
export CUDNN_PATH=/usr
export PATH="/usr/local/cuda-12/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export MAX_JOBS=4
export NVTE_FRAMEWORK=pytorch
# Default configuration
MODEL_NAME="${1:-HuggingFaceTB/SmolLM2-135M-Instruct}"
NUM_GPUS="${2:-4}"  # Use all 4 GPUs by default
GPU_MEMORY_UTIL="${3:-0.4}"  # 40% per GPU (leave room for training)

# vLLM LoRA configuration
MAX_LORAS="${MAX_LORAS:-8}"  # Support up to 8 concurrent LoRA adapters
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

# Training backend (optional)
TRAINING_BACKEND="${TRAINING_BACKEND:-megatron-bridge}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Tinkerbell Multi-GPU Server Startup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. GPU support required.${NC}"
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ Found ${AVAILABLE_GPUS} GPUs${NC}"

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo -e "${RED}ERROR: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available${NC}"
    exit 1
fi

# Show GPU info
echo ""
echo -e "${YELLOW}GPU Configuration:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -n "$NUM_GPUS" | while IFS=, read -r idx name mem; do
    echo -e "  GPU ${idx}: ${name} (${mem})"
done

# Configuration summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  Model:                  ${GREEN}${MODEL_NAME}${NC}"
echo -e "  Tensor Parallel Size:   ${GREEN}${NUM_GPUS}${NC} GPUs"
echo -e "  GPU Memory Util:        ${GREEN}${GPU_MEMORY_UTIL}${NC} (per GPU)"
echo -e "  Max LoRA Adapters:      ${GREEN}${MAX_LORAS}${NC}"
echo -e "  Max LoRA Rank:          ${GREEN}${MAX_LORA_RANK}${NC}"
echo -e "  Training Backend:       ${GREEN}${TRAINING_BACKEND}${NC}"
echo ""

# Memory calculation
if command -v bc &> /dev/null; then
    TOTAL_GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n "$NUM_GPUS" | awk '{sum+=$1} END {print sum}')
    VLLM_MEMORY=$(echo "$TOTAL_GPU_MEMORY * $GPU_MEMORY_UTIL" | bc)
    echo -e "  Total GPU Memory:       ${BLUE}${TOTAL_GPU_MEMORY}${NC} MiB (across ${NUM_GPUS} GPUs)"
    echo -e "  vLLM Will Use:          ${BLUE}~${VLLM_MEMORY}${NC} MiB (${GPU_MEMORY_UTIL} × ${TOTAL_GPU_MEMORY})"
    TRAINING_MEMORY=$(echo "$TOTAL_GPU_MEMORY - $VLLM_MEMORY" | bc)
    echo -e "  Available for Training: ${BLUE}~${TRAINING_MEMORY}${NC} MiB"
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if port is already in use
PORT=8000
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}ERROR: Port ${PORT} is already in use${NC}"
    echo "Kill the existing process or use a different port"
    exit 1
fi

# Set environment variables for vLLM
export VLLM_MODEL="$MODEL_NAME"
export VLLM_TENSOR_PARALLEL_SIZE="$NUM_GPUS"
export VLLM_GPU_MEMORY_UTIL="$GPU_MEMORY_UTIL"
export VLLM_ENABLE_LORA="true"
export VLLM_MAX_LORAS="$MAX_LORAS"
export VLLM_MAX_LORA_RANK="$MAX_LORA_RANK"
export VLLM_DTYPE="auto"
export VLLM_TRUST_REMOTE_CODE="true"

# Optional: Set CUDA_VISIBLE_DEVICES if you want to use specific GPUs
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Set Megatron-Bridge training backend
if [ "$TRAINING_BACKEND" == "megatron-bridge" ]; then
    export USE_MEGATRON="true"
    export MEGATRON_MODEL="$MODEL_NAME"
    export USE_VLLM="true"
    echo -e "${GREEN}✓ Megatron-Bridge backend enabled${NC}"
fi

echo -e "${YELLOW}Starting server...${NC}"
echo ""
echo -e "${BLUE}Environment variables set:${NC}"
echo "  VLLM_MODEL=$VLLM_MODEL"
echo "  VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE"
echo "  VLLM_GPU_MEMORY_UTIL=$VLLM_GPU_MEMORY_UTIL"
echo "  VLLM_ENABLE_LORA=$VLLM_ENABLE_LORA"
echo "  VLLM_MAX_LORAS=$VLLM_MAX_LORAS"
echo "  VLLM_MAX_LORA_RANK=$VLLM_MAX_LORA_RANK"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Starting Tinkerbell server on port ${PORT}...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Server logs will appear below. Press ${YELLOW}Ctrl+C${NC} to stop."
echo ""

# Start the server
python3.12 src/app.py
