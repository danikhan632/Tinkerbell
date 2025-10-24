# Tink Worker Configuration Guide

## Backend Architecture

The Tink worker supports multiple backends that can work together:

```
┌─────────────────────────────────────────────────┐
│                  Tink Worker                    │
├─────────────────────────────────────────────────┤
│  Training (fwdbwd/optim)  │  Sampling (sample)  │
│         Megatron          │        vLLM         │
└─────────────────────────────────────────────────┘
```

## Configuration Modes

### Mode 1: Megatron Training + vLLM Sampling (Recommended)

This mode uses Megatron for training and vLLM for high-performance sampling with LoRA adapters.

```bash
export USE_MEGATRON=true
export USE_VLLM=true

# Megatron server for training
export MEGATRON_SERVER_URL=http://localhost:5000

# vLLM server for sampling
export VLLM_BASE_URL=http://localhost:8000
export VLLM_ENABLE_LORA=true
export VLLM_MAX_LORAS=4
```

**Start services:**
```bash
# Terminal 1: Start Megatron training server
# (Start your Megatron server according to your setup)

# Terminal 2: Start vLLM server with LoRA support
trl vllm-serve \
    --model meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --port 8000

# Terminal 3: Start Tink worker
python src/app.py
```

### Mode 2: Megatron Only (Training + Sampling)

This mode uses Megatron for both training and sampling.

```bash
export USE_MEGATRON=true
export USE_VLLM=false

# Megatron server for both training and sampling
export MEGATRON_SERVER_URL=http://localhost:5000
```

**Start services:**
```bash
# Terminal 1: Start Megatron server
# (Your Megatron server must support both training and sampling)

# Terminal 2: Start Tink worker
python src/app.py
```

### Mode 3: vLLM Sampling Only

This mode uses vLLM for sampling only (no training support).

```bash
export USE_MEGATRON=false
export USE_VLLM=true

# vLLM server for sampling
export VLLM_BASE_URL=http://localhost:8000
```

**Note:** Training operations (fwdbwd, optim, add_lora) will fail in this mode.

## Environment Variables Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MEGATRON` | `false` | Enable Megatron backend for training |
| `USE_VLLM` | `false` | Enable vLLM backend for sampling |
| `MAX_WORKERS` | `4` | Number of worker threads |

### Megatron Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEGATRON_SERVER_URL` | `http://localhost:5000` | Megatron server URL |

### vLLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | - | vLLM server base URL (e.g., `http://localhost:8000`) |
| `VLLM_HOST` | `0.0.0.0` | vLLM server host (used if BASE_URL not set) |
| `VLLM_PORT` | `8000` | vLLM server port (used if BASE_URL not set) |
| `VLLM_GROUP_PORT` | `51216` | Port for weight update group |
| `VLLM_GPU_MEMORY_UTIL` | `0.2` | GPU memory fraction (5% - leaves 95% for training) |
| `VLLM_ENABLE_LORA` | `true` | Enable LoRA adapter support |
| `VLLM_MAX_LORAS` | `4` | Maximum concurrent LoRA adapters |
| `VLLM_MAX_LORA_RANK` | `64` | Maximum LoRA rank |

## Operation Support by Backend

| Operation | Megatron | vLLM | Notes |
|-----------|----------|------|-------|
| `sample` | ✓ | ✓ | Sampling/inference |
| `asample` | ✓ | ✓ | Async sampling |
| `fwdbwd` | ✓ | ✗ | Forward-backward pass |
| `optim` | ✓ | ✗ | Optimizer step |
| `add_lora` | ✓ | ✗ | Create LoRA adapter |
| `remove_lora` | ✓ | ✗ | Delete LoRA adapter |

## Routing Logic

The worker uses the following routing logic:

```
if USE_MEGATRON:
    if operation in [sample, asample]:
        if USE_VLLM:
            → vLLM (fast sampling with LoRA)
        else:
            → Megatron server (sampling)
    elif operation in [fwdbwd, optim, add_lora, ...]:
        → Megatron (training)
else:
    if operation in [sample, asample]:
        if USE_VLLM:
            → vLLM (sampling only)
        else:
            → Error (no backend available)
    else:
        → Error (training requires Megatron)
```

## Troubleshooting

### Error: "No valid sampling backend configured"

**Cause:** Neither `USE_MEGATRON` nor `USE_VLLM` is enabled.

**Solution:**
```bash
export USE_VLLM=true  # For sampling
# OR
export USE_MEGATRON=true  # For sampling + training
```

### Error: "Training not supported in non-Megatron mode"

**Cause:** Trying to train without Megatron enabled.

**Solution:**
```bash
export USE_MEGATRON=true
```

### Error: "vLLM server can't be reached"

**Cause:** vLLM server is not running or URL is incorrect.

**Solution:**
```bash
# Check if vLLM server is running
curl http://localhost:8000/health/

# Start vLLM server if not running
trl vllm-serve --model your-model --enable-lora
```

### Error: "Gradient computation error" (in-place operation)

**Cause:** Concurrent training operations interfering with each other in Megatron mode.

**Solution:**
- Megatron mode processes jobs sequentially to avoid this
- Ensure `USE_MEGATRON=true` is set
- Only one training job runs at a time in Megatron mode

## Example Configurations

### For Testing (concurrent users test)

```bash
# Use HuggingFace backend (not Megatron) for concurrent training
export USE_MEGATRON=false
export USE_VLLM=false
export MAX_WORKERS=4

python src/test_concurrent_users.py 4
```

### For Production (Megatron + vLLM)

```bash
export USE_MEGATRON=true
export USE_VLLM=true
export MEGATRON_SERVER_URL=http://localhost:5000
export VLLM_BASE_URL=http://localhost:8000
export VLLM_ENABLE_LORA=true
export MAX_WORKERS=4

python src/app.py
```
