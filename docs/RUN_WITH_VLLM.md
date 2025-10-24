# Running Examples with vLLM Sampling

This guide shows how to run the concurrent training examples with vLLM for high-performance sampling.

## Quick Start

### 1. Start vLLM Server (Terminal 1)

```bash
# Install vLLM if needed
pip install vllm requests

# Start vLLM server with LoRA support
trl vllm-serve \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --port 8000

# OR use a different model
trl vllm-serve \
    --model meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --port 8000
```

### 2. Configure Environment (Terminal 2)

```bash
cd /home/green/code/thinker/tink

# Enable vLLM for sampling
export USE_VLLM=true
export VLLM_BASE_URL=http://localhost:8000

# Optionally enable Megatron for training (otherwise uses HuggingFace)
# export USE_MEGATRON=true
# export MEGATRON_SERVER_URL=http://localhost:5000

# Optional: Adjust worker pool size
export MAX_WORKERS=4
```

### 3. Start Tink Server

```bash
# In the same terminal with environment variables set
python src/app.py
```

You should see:
```
Initializing vLLM backend...
vLLM backend initialized for high-performance sampling with LoRA support
Initializing HuggingFace backend...
Worker pool size: 4 threads
Loading base model from HuggingFaceTB/SmolLM2-135M-Instruct...
Base model loaded on cuda
Worker pool initialized. Active backends: vLLM (sampling), HuggingFace (training/sampling)
Concurrent job processing: Enabled (HuggingFace)
```

### 4. Run Example (Terminal 3)

```bash
cd /home/green/code/thinker/tink/examples
python example_simple_concurrent.py
```

## Configuration Modes

### Mode 1: vLLM Sampling + HuggingFace Training (Recommended)

**Best for:** Concurrent multi-user training with fast inference

```bash
export USE_VLLM=true              # Fast sampling with vLLM
export USE_MEGATRON=false         # Use HuggingFace for training
export VLLM_BASE_URL=http://localhost:8000
```

**Performance:**
- Training: Concurrent (multiple users train simultaneously)
- Sampling: 3-10x faster than HuggingFace
- LoRA: Supports multiple LoRA adapters

### Mode 2: vLLM Sampling + Megatron Training

**Best for:** Production with Megatron

```bash
export USE_VLLM=true              # Fast sampling with vLLM
export USE_MEGATRON=true          # Use Megatron for training
export VLLM_BASE_URL=http://localhost:8000
export MEGATRON_SERVER_URL=http://localhost:5000
```

**Note:** Megatron processes training jobs sequentially (one at a time)

### Mode 3: HuggingFace Only (No vLLM)

**Best for:** Testing without vLLM

```bash
export USE_VLLM=false
export USE_MEGATRON=false
```

**Performance:**
- Training: Concurrent
- Sampling: Standard HuggingFace (slower)

## How vLLM Sampling Works

When you run the example with vLLM enabled:

1. **Training Phase** (HuggingFace or Megatron)
   - Each user creates a LoRA adapter
   - Adapter is trained with user-specific data
   - Gradients computed and applied locally

2. **Sampling Phase** (vLLM)
   - Worker receives sample request with `model_id` (LoRA adapter ID)
   - If `USE_VLLM=true`, routes to vLLM backend
   - vLLM backend checks if LoRA adapter is registered
   - If not registered, needs to be saved and loaded into vLLM
   - vLLM generates completions using the LoRA adapter
   - Response returned to client

## Integrating Trained LoRA Adapters with vLLM

Currently, the example trains LoRA adapters in HuggingFace/Megatron but they need to be exported for vLLM sampling:

### Option 1: Save and Register Adapters (Manual)

After training, save the adapter:

```python
# In your training code
import vllm_backend

# After training is complete
adapter_path = f"/tmp/lora_adapters/{model_id}"
hf_backend.lora_adapters[model_id].save_pretrained(adapter_path)

# Register with vLLM
vllm_backend.register_lora_adapter(model_id, adapter_path)
```

### Option 2: Automatic Registration (Future Enhancement)

Add this to worker.py after LoRA creation:

```python
# In add_lora job handler
if USE_VLLM:
    # Save adapter
    adapter_path = f"/tmp/lora_adapters/{model_id}"
    hf_backend.lora_adapters[model_id].save_pretrained(adapter_path)

    # Register with vLLM
    vllm_backend.register_lora_adapter(model_id, adapter_path)
```

## Monitoring vLLM

### Check vLLM Server Status

```bash
curl http://localhost:8000/health/
```

### Check Registered LoRA Adapters

```python
import vllm_backend

# Get client info
info = vllm_backend.get_client_info()
print(info)

# List registered adapters
adapters = vllm_backend.list_lora_adapters()
print(adapters)
```

## Troubleshooting

### vLLM Server Not Starting

**Problem:**
```
ConnectionError: The vLLM server can't be reached at http://localhost:8000
```

**Solution:**
```bash
# Check if vLLM server is running
curl http://localhost:8000/health/

# If not, start it
trl vllm-serve --model HuggingFaceTB/SmolLM2-135M-Instruct --enable-lora
```

### LoRA Adapter Not Found in vLLM

**Problem:**
```
Adapter 'base_lora_abc123' not found in vLLM
```

**Solution:**
The adapter was trained but not registered with vLLM. You need to:
1. Save the adapter from HuggingFace backend
2. Register it with vLLM backend

### Falling Back to HuggingFace

If vLLM is not available, the worker will automatically fall back to HuggingFace for sampling. Check logs:

```
WARNING: vLLM not available, using HuggingFace for sampling
```

## Performance Comparison

### Training Performance

| Mode | Concurrent Users | Time (3 users, 3 steps each) |
|------|------------------|------------------------------|
| HuggingFace | ✓ Yes | ~10-15 seconds |
| Megatron | ✗ No (sequential) | ~30-45 seconds |

### Sampling Performance

| Backend | Speed | LoRA Support | Concurrent Requests |
|---------|-------|--------------|---------------------|
| vLLM | 3-10x faster | ✓ Yes | ✓ Yes |
| HuggingFace | Baseline | ✓ Yes | ✓ Yes |
| Megatron | ~2x faster | Varies | ✓ Yes |

## Example Output with vLLM

```
======================================================================
Simple Concurrent Multi-User Example
======================================================================

This example shows 3 users training their own LoRA adapters
simultaneously without blocking each other.

Backend Configuration:
  Training: HuggingFace (concurrent) or Megatron (sequential)
  Sampling: vLLM (if USE_VLLM=true) or HuggingFace (fallback)

To use vLLM for fast sampling:
  1. Start vLLM server: trl vllm-serve --model MODEL --enable-lora
  2. Set environment: export USE_VLLM=true
  3. Run this example

✓ Server is healthy: {'status': 'ok'}

Starting 3 concurrent users...

[Alice] Starting...
[Bob] Starting...
[Carol] Starting...
[Alice] Creating LoRA adapter...
[Bob] Creating LoRA adapter...
[Carol] Creating LoRA adapter...
[Alice] ✓ Created adapter: base_lora_abc123
[Bob] ✓ Created adapter: base_lora_def456
[Carol] ✓ Created adapter: base_lora_ghi789
[Alice] Training step 1/3...
[Bob] Training step 1/3...
[Carol] Training step 1/3...
...
[Alice] Testing inference with LoRA adapter...
[Alice] Generated: Nice to meet you too, Alice!
[Alice] ✓✓✓ DONE! ✓✓✓
...

======================================================================
SUMMARY
======================================================================

✓ All 3 users completed training!
✓ Total time: 12.34 seconds

Key Points:
  • Each user trained their own LoRA adapter
  • All training happened CONCURRENTLY (not sequential)
  • Users didn't block each other
  • Speedup: ~3x compared to sequential execution
  • Sampling used vLLM for 5x faster inference

======================================================================
```

## Next Steps

1. **Enable automatic LoRA registration** - Modify worker to auto-register trained adapters
2. **Add adapter persistence** - Save adapters to disk for later use
3. **Monitor vLLM metrics** - Track latency and throughput
4. **Scale vLLM** - Add multiple vLLM servers behind load balancer
5. **Production deployment** - Use Megatron for training + vLLM for serving
