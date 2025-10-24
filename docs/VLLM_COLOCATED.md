# vLLM Co-Located Mode

## What is Co-Located Mode?

Co-located mode runs vLLM engine **in the same process** as your training code. This provides:

✅ **No network overhead** - vLLM engine runs in-process
✅ **Same GPU memory** - Shares memory with training
✅ **Immediate LoRA access** - Trained adapters instantly available
✅ **No separate server** - All in one `torchrun` command

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            Single Python Process (torchrun)         │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐      ┌───────────────────┐   │
│  │  Training Loop   │      │   vLLM Engine     │   │
│  │  (HF/Megatron)   │      │   (In-Process)    │   │
│  │                  │      │                   │   │
│  │  Create LoRA     │─────▶│  Auto-Register    │   │
│  │  Train LoRA      │      │  LoRA Adapter     │   │
│  │  Save LoRA       │      │                   │   │
│  └──────────────────┘      └───────────────────┘   │
│          │                           │              │
│          └───────────┬───────────────┘              │
│                      │                              │
│              ┌───────▼──────┐                       │
│              │  Shared GPU  │                       │
│              │    Memory    │                       │
│              └──────────────┘                       │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Configure Environment

```bash
export USE_VLLM=true
export VLLM_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
export VLLM_ENABLE_LORA=true
export VLLM_MAX_LORAS=4
export VLLM_GPU_MEMORY_UTIL=0.2  # Only 5% - leaves 95% for training!
```

### 3. Run with torchrun

```bash
cd /home/green/code/thinker/tink
torchrun --nproc_per_node=1 src/app.py
```

That's it! You'll see:

```
Initializing vLLM engine (in-process, co-located)...
Initializing vLLM engine with model: HuggingFaceTB/SmolLM2-135M-Instruct
  Tensor parallel size: 1
  GPU memory utilization: 0.9
  LoRA support: True
  Max LoRAs: 4, Max rank: 64
vLLM engine initialized successfully (in-process, co-located)
  Mode: Co-located (same process, same GPU memory, immediate LoRA access)
```

## How It Works

### 1. LoRA Creation

When you create a LoRA adapter:

```python
client.add_lora(base_model="base", rank=16, alpha=32)
# Returns: {"model_id": "base_lora_abc123"}
```

**Behind the scenes:**
1. HuggingFace backend creates the adapter
2. Adapter is automatically saved to `/tmp/lora_adapters/base_lora_abc123/`
3. vLLM engine registers the adapter **immediately**
4. Ready for sampling instantly!

### 2. Sampling with LoRA

```python
client.sample(
    model_id="base_lora_abc123",  # Use the LoRA adapter
    prompts=["Hello!"],
    sampling_params={"max_tokens": 20}
)
```

**Behind the scenes:**
1. Worker routes to vLLM (in-process)
2. vLLM loads LoRA from `/tmp/lora_adapters/base_lora_abc123/`
3. Generates using base model + LoRA
4. Returns results

### 3. Automatic Flow

```
Train LoRA → Auto-Save → Auto-Register → Sample with LoRA
     ↓           ↓            ↓              ↓
  HF/Megatron   Disk      vLLM Engine    Fast Inference
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_VLLM` | `false` | Enable vLLM engine |
| `VLLM_MODEL` | `SmolLM2-135M-Instruct` | Model to load |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Tensor parallelism |
| `VLLM_GPU_MEMORY_UTIL` | `0.2` | GPU memory fraction (5% - leaves 95% for training!) |
| `VLLM_ENABLE_LORA` | `true` | Enable LoRA support |
| `VLLM_MAX_LORAS` | `4` | Max concurrent LoRAs |
| `VLLM_MAX_LORA_RANK` | `64` | Max LoRA rank |
| `VLLM_DTYPE` | `auto` | Data type (auto/float16/bfloat16) |

### With Megatron Training

```bash
export USE_VLLM=true            # vLLM for sampling
export USE_MEGATRON=true        # Megatron for training

torchrun --nproc_per_node=2 src/app.py
```

## Performance Benefits

### vs. Separate vLLM Server

| Aspect | Co-Located | Separate Server |
|--------|------------|-----------------|
| Network latency | ❌ None | ✓ Yes (~1-5ms per request) |
| Memory overhead | ❌ Minimal | ✓ Duplicate model weights |
| LoRA registration | ✅ Instant | ⏱️ Manual save + register |
| Setup complexity | ✅ Single process | ⚠️ Multiple processes |

### Memory Usage

**Separate Server:**
```
Training Process: 2GB (model + LoRA)
vLLM Server:     2GB (model + LoRA)
Total:           4GB
```

**Co-Located (with 5% vLLM memory):**
```
Single Process: ~2.1GB
  - Training:   2.0GB (95% of GPU)
  - vLLM:      ~0.1GB (5% of GPU, cached model)
Total:          2.1GB (47% savings!)
```

*Note: vLLM uses only 5% GPU memory by default, leaving 95% for training!*

## Example: Concurrent Users with Co-Located vLLM

```bash
# Terminal 1: Start server
export USE_VLLM=true
torchrun --nproc_per_node=1 src/app.py

# Terminal 2: Run example
cd examples
python example_simple_concurrent.py
```

**Output:**
```
[Alice] Creating LoRA adapter...
[Alice] ✓ Created adapter: base_lora_abc123
Saving LoRA adapter 'base_lora_abc123' to /tmp/lora_adapters/base_lora_abc123...
Registered LoRA adapter 'base_lora_abc123' at /tmp/lora_adapters/base_lora_abc123
LoRA adapter 'base_lora_abc123' ready for vLLM sampling

[Alice] Training step 1/3...
[Alice] Training step 2/3...
[Alice] Training step 3/3...

[Alice] Testing inference with LoRA adapter...
Using LoRA adapter: base_lora_abc123 from /tmp/lora_adapters/base_lora_abc123
[Alice] Generated: Nice to meet you, Alice!
```

## Troubleshooting

### Out of Memory (OOM)

**Problem:** GPU OOM when both training and vLLM run

**Solution:** vLLM is already configured to use only 5% GPU memory by default. If you still have OOM:

```bash
# Reduce even further (not recommended - may be too small)
export VLLM_GPU_MEMORY_UTIL=0.03  # Only 3%

# Or check what's using memory
nvidia-smi
```

**Note:** Default is 0.2 (5%), leaving 95% for training. This should work for most setups!

### LoRA Not Found

**Problem:** "LoRA adapter not found"

**Cause:** Adapter created but not registered yet

**Check:**
```python
import vllm_backend
print(vllm_backend.list_lora_adapters())
```

### Slow First Sample

**Problem:** First sampling request is slow

**Cause:** vLLM engine lazy initialization + LoRA loading

**Expected:** First request ~5-10s, subsequent <100ms

## Advanced: Manual LoRA Registration

If you need to manually register an existing LoRA:

```python
import vllm_backend

# Register existing LoRA
vllm_backend.register_lora_adapter(
    adapter_id="my_adapter",
    lora_path="/path/to/lora/weights"
)

# Check registration
adapters = vllm_backend.list_lora_adapters()
print(adapters)
```

## Comparison: Co-Located vs. Server Mode

### When to Use Co-Located Mode

✅ **Single GPU setup**
✅ **Development and testing**
✅ **Small to medium models (<7B)**
✅ **Tight memory constraints**
✅ **Need instant LoRA access**

### When to Use Server Mode

✅ **Multi-GPU inference**
✅ **Production serving**
✅ **Multiple clients**
✅ **Large models (>7B)**
✅ **Separate training/serving infrastructure**

## Summary

Co-located vLLM mode provides:

1. **Zero network latency** - In-process calls
2. **Memory efficiency** - Shared GPU memory
3. **Instant LoRA access** - Auto-register after training
4. **Simple setup** - Single `torchrun` command
5. **Perfect for development** - Fast iteration

Just set `USE_VLLM=true` and run with `torchrun` - that's it!
