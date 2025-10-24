# Running Modes for Tinker API Server

## Quick Start

### Mode 1: vLLM-Only (Inference Only, Recommended)
Fast inference with LoRA adapters, no training capability.

```bash
./start_vllm_only.sh
```

**Pros:**
- Simple setup (no torchrun needed)
- Fast inference with vLLM
- LoRA adapter support for inference
- Low GPU memory usage (configurable)

**Cons:**
- No training capability
- Can't create new LoRA adapters (only use pre-trained ones)

---

### Mode 2: HuggingFace Only (Training + Inference)
Full training and inference using HuggingFace transformers.

```bash
export USE_VLLM=false
export USE_MEGATRON=false
python src/app.py
```

**Pros:**
- Full training support
- Can create and train LoRA adapters
- Simple, no distributed setup needed

**Cons:**
- Slower inference than vLLM
- Higher memory usage

---

### Mode 3: HuggingFace + vLLM (Training + Fast Inference)
Training with HuggingFace, inference with vLLM.

```bash
export USE_VLLM=true
export USE_MEGATRON=false
export VLLM_GPU_MEMORY_UTIL=0.3  # Reserve 30% for vLLM, 70% for training
python src/app.py
```

**Pros:**
- Full training support
- Fast inference with vLLM
- Can train LoRAs and immediately use them for inference

**Cons:**
- Higher GPU memory usage (both models loaded)
- Need to carefully balance memory allocation

---

### Mode 4: Megatron + vLLM (Distributed Training + Fast Inference)
Advanced: Distributed training with Megatron-LM, inference with vLLM.

```bash
export USE_VLLM=true
export USE_MEGATRON=true
export VLLM_GPU_MEMORY_UTIL=0.1  # Reserve 10% for vLLM
torchrun --nproc_per_node=2 --standalone src/app.py
```

**Pros:**
- Distributed training across multiple GPUs
- Fast inference with vLLM
- Scale to larger models

**Cons:**
- Complex setup
- Requires proper distributed initialization
- Higher GPU memory requirements

**Important:** Currently has initialization conflicts - needs fixes!

---

## Environment Variables

### vLLM Configuration
```bash
export VLLM_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
export VLLM_GPU_MEMORY_UTIL=0.2  # Fraction of GPU memory (0.0-1.0)
export VLLM_ENABLE_LORA=true
export VLLM_MAX_LORAS=4
export VLLM_MAX_LORA_RANK=64
export VLLM_TENSOR_PARALLEL_SIZE=1
```

### Backend Selection
```bash
export USE_VLLM=true          # Enable vLLM for inference
export USE_MEGATRON=false     # Enable Megatron for training
```

### Server Configuration
```bash
export PORT=8000              # Flask server port
export MAX_WORKERS=4          # Thread pool size for HuggingFace backend
```

---

## Troubleshooting

### Issue: "Infinite loop" or processes keep spawning
**Cause:** Worker thread starting in vLLM spawned child processes
**Fix:** Use the updated code with multiprocessing guards, or run vLLM-only mode

### Issue: TCP timeout errors with Megatron + vLLM
**Cause:** Distributed initialization conflicts between torchrun, Megatron, and vLLM
**Fix:** Currently being worked on. Use Mode 1, 2, or 3 instead.

### Issue: OOM (Out of Memory)
**Cause:** Both HuggingFace and vLLM loading models into GPU memory
**Fix:**
- Use vLLM-only mode if you don't need training
- Reduce `VLLM_GPU_MEMORY_UTIL` to reserve less memory for vLLM
- Use smaller models

### Issue: "CUDA capability sm_61 is not compatible"
**Cause:** Tesla P40 GPU is too old for current PyTorch build
**Fix:** This is a warning and may still work, but for best results install PyTorch with CUDA 11.8 or earlier
