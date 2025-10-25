# 🚀 Tinkerbell Multi-GPU Quick Start

## Your System
- **GPUs Available**: 4
- **Python**: 3.12
- **Backend**: Megatron-Bridge + vLLM

## One-Line Commands

### Start Server (All 4 GPUs)
```bash
cd /home/shadeform/Tinkerbell && ./scripts/run_server_4gpu.sh
```

### Start Server (2 GPUs)
```bash
cd /home/shadeform/Tinkerbell && ./scripts/run_server_2gpu.sh
```

### Custom Model/Config
```bash
cd /home/shadeform/Tinkerbell && ./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 4 0.5
```

## Test the Server

### In Another Terminal
```bash
cd /home/shadeform/Tinkerbell
python3.12 examples/example_simple_concurrent.py
```

## Monitor GPUs
```bash
watch -n 1 nvidia-smi
```

## Stop Server
```bash
# Press Ctrl+C in server terminal
# OR
lsof -ti:8000 | xargs kill -9
```

## Configuration Files
- **Full Guide**: `MULTIGPU_SETUP.md`
- **Migration**: `MEGATRON_BRIDGE_MIGRATION.md`
- **Changes**: `CHANGES_SUMMARY.md`

## Common Issues

### "Port already in use"
```bash
lsof -ti:8000 | xargs kill -9
```

### "Out of memory"
```bash
# Use fewer GPUs or less memory
./scripts/run_server_multigpu.sh YourModel 2 0.3
```

### Check if vLLM works
```bash
python3.12 -c "import vllm; print('✓ vLLM available:', vllm.__version__)"
```

## Expected Performance

| Setup | Inference Speed | Memory | Training Capacity |
|-------|----------------|--------|-------------------|
| 1 GPU | ~50 tok/s | 24GB | Low |
| 2 GPUs | ~90 tok/s | 48GB | Medium |
| 4 GPUs | ~160 tok/s | 96GB | High |

## Architecture

```
┌────────────────┐
│ Flask/Celery   │
└────────┬───────┘
         │
    ┌────┴─────┐
    │          │
┌───▼──┐   ┌──▼────────┐
│ vLLM │   │ Megatron- │
│ 4GPU │◄─►│  Bridge   │
└──────┘   └───────────┘
   │              │
   └──In-Memory───┘
      Streaming
```

## What Happens When You Run

1. **vLLM Initializes** across 4 GPUs (tensor parallelism)
2. **Model Loads** split across GPUs (~25% per GPU)
3. **Server Starts** on port 8000
4. **Ready** for training + inference!

## Examples

```bash
# Production (all GPUs, balanced)
./scripts/run_server_4gpu.sh

# Development (fewer GPUs)
./scripts/run_server_2gpu.sh

# Large model
./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 4 0.4

# Small model, max training
export MAX_LORAS=4
./scripts/run_server_multigpu.sh HuggingFaceTB/SmolLM2-135M-Instruct 2 0.2
```

## Need Help?

1. Read: `MULTIGPU_SETUP.md` (full guide)
2. Read: `MEGATRON_BRIDGE_MIGRATION.md` (backend details)
3. Check: Server logs for errors
4. Test: `python3.12 examples/example_bridge_weight_streaming.py`

---

**Ready? Run this:**
```bash
cd /home/shadeform/Tinkerbell && ./scripts/run_server_4gpu.sh
```
