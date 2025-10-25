# Multi-GPU vLLM Setup Guide

## Quick Start

You have **4 GPUs** available. Here are the commands to run the Tinkerbell server with multi-GPU vLLM:

### 🚀 Option 1: Use All 4 GPUs (Recommended)
```bash
cd /home/shadeform/Tinkerbell
./scripts/run_server_4gpu.sh
```

### 🚀 Option 2: Use 2 GPUs
```bash
cd /home/shadeform/Tinkerbell
./scripts/run_server_2gpu.sh
```

### 🚀 Option 3: Custom Configuration
```bash
cd /home/shadeform/Tinkerbell
./scripts/run_server_multigpu.sh [model_name] [num_gpus] [gpu_memory_util]

# Examples:
./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 4 0.5
./scripts/run_server_multigpu.sh Qwen/Qwen3-0.6B 2 0.3
```

## 📋 Configuration Options

### Script Parameters

```bash
./scripts/run_server_multigpu.sh <MODEL> <NUM_GPUS> <GPU_MEMORY_UTIL>
```

| Parameter | Default | Description | Examples |
|-----------|---------|-------------|----------|
| `MODEL` | SmolLM2-135M-Instruct | HuggingFace model name | `meta-llama/Llama-3.2-1B`, `Qwen/Qwen3-0.6B` |
| `NUM_GPUS` | 4 | Number of GPUs for tensor parallelism | 1, 2, 4 |
| `GPU_MEMORY_UTIL` | 0.4 | Memory fraction per GPU (for vLLM) | 0.2-0.8 |

### Environment Variables (Advanced)

Set these before running the script for more control:

```bash
# LoRA Configuration
export MAX_LORAS=8              # Max concurrent LoRA adapters
export MAX_LORA_RANK=64         # Max LoRA rank

# Training Backend
export TRAINING_BACKEND="megatron-bridge"  # or "huggingface"

# GPU Selection (optional)
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Use specific GPUs

# Then run:
./scripts/run_server_multigpu.sh
```

## 🎯 How vLLM Multi-GPU Works

### Tensor Parallelism
vLLM uses **tensor parallelism** to distribute model weights across GPUs:

```
┌─────────────────────────────────────────────┐
│          Single Model (e.g., 1B params)     │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │  Tensor Parallelism    │
        └───────────┬────────────┘
                    │
    ┌───────┬───────┼───────┬───────┐
    │       │       │       │       │
    ▼       ▼       ▼       ▼       ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│GPU 0 ││GPU 1 ││GPU 2 ││GPU 3 │
│ 25%  ││ 25%  ││ 25%  ││ 25%  │  ← Model weights split
└──────┘└──────┘└──────┘└──────┘
```

**Benefits:**
- Load larger models that don't fit on 1 GPU
- Faster inference through parallel computation
- Each GPU only needs a fraction of model memory

## 💡 Memory Allocation Strategy

### Default Configuration (4 GPUs @ 0.4 utilization)

Assuming 24GB per GPU (adjust for your GPUs):

```
Per GPU:
├─ Total Memory:        24 GB
├─ vLLM (40%):          9.6 GB  ← For inference
└─ Training (60%):      14.4 GB ← For Megatron-Bridge training

Across 4 GPUs:
├─ Total Memory:        96 GB
├─ vLLM Total:          38.4 GB ← Distributed model
└─ Training Total:      57.6 GB ← LoRA training
```

### Recommended Memory Splits

| Use Case | GPU Util | vLLM | Training | Notes |
|----------|----------|------|----------|-------|
| **Inference Heavy** | 0.7 | 70% | 30% | More concurrent requests |
| **Balanced** | 0.4-0.5 | 40-50% | 50-60% | Default, good for both |
| **Training Heavy** | 0.2-0.3 | 20-30% | 70-80% | More training capacity |

## 🔧 Configuration Examples

### Example 1: Large Model (Llama 3.2-1B) on 4 GPUs
```bash
# Use more GPUs for distribution, moderate memory for training
./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 4 0.4
```

### Example 2: Small Model (SmolLM) on 2 GPUs
```bash
# Fewer GPUs, more memory for training
./scripts/run_server_multigpu.sh HuggingFaceTB/SmolLM2-135M-Instruct 2 0.3
```

### Example 3: Maximum Inference Performance
```bash
# All GPUs, high memory utilization
./scripts/run_server_multigpu.sh Qwen/Qwen3-0.6B 4 0.7
```

### Example 4: Maximum Training Capacity
```bash
# Minimal vLLM memory, max for training
./scripts/run_server_multigpu.sh HuggingFaceTB/SmolLM2-135M-Instruct 2 0.2
```

## 📊 Performance Expectations

### Inference Speed (vLLM)

With tensor parallelism across 4 GPUs:
- **Single GPU**: ~50 tokens/sec
- **2 GPUs (TP=2)**: ~90 tokens/sec (1.8x)
- **4 GPUs (TP=4)**: ~160 tokens/sec (3.2x)

*Actual speedup depends on model size and communication overhead*

### Training with Megatron-Bridge

Multi-GPU setup allows:
- **Concurrent Users**: 3-10 users training simultaneously
- **Larger Batches**: More memory = bigger batches
- **Faster Training**: Megatron's distributed parallelism

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce GPU memory utilization
```bash
./scripts/run_server_multigpu.sh YourModel 4 0.3  # Try 0.3 instead of 0.4
```

**Solution 2**: Use more GPUs (distribute the load)
```bash
./scripts/run_server_multigpu.sh YourModel 4 0.4  # Use all 4 GPUs
```

**Solution 3**: Reduce max LoRAs
```bash
export MAX_LORAS=4  # Instead of default 8
./scripts/run_server_multigpu.sh YourModel 4 0.4
```

### Issue: "Port 8000 already in use"

**Solution**: Kill existing server
```bash
lsof -ti:8000 | xargs kill -9
# Then restart
./scripts/run_server_4gpu.sh
```

### Issue: "nvidia-smi not found"

**Solution**: Install NVIDIA drivers
```bash
# Check if GPUs are detected
lspci | grep -i nvidia

# Install drivers (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-driver-535  # Or latest version
sudo reboot
```

### Issue: "vLLM initialization failed"

**Check 1**: Verify vLLM is installed
```bash
python -c "import vllm; print(vllm.__version__)"
```

**Check 2**: Verify GPU compatibility
```bash
nvidia-smi  # Should show all 4 GPUs
```

**Check 3**: Check logs
```bash
# The server will print detailed vLLM initialization logs
# Look for errors in the output
```

### Issue: Slow multi-GPU performance

**Possible causes:**
1. **GPU communication bottleneck**: Check if GPUs are on same node with NVLink
   ```bash
   nvidia-smi topo -m  # Should show NVLink connections
   ```

2. **Network overhead**: For distributed setup, ensure high-bandwidth interconnect

3. **Model too small**: Tensor parallelism overhead > benefits for very small models
   - Use TP=1 or TP=2 for models < 500M parameters

## 🔍 Monitoring

### Check GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed per-GPU stats
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

### Check vLLM Status
```bash
# Server logs show vLLM initialization
# Look for:
# "Initializing vLLM engine with model: ..."
# "Tensor parallel size: 4"
# "vLLM engine initialized successfully"
```

### Test Multi-GPU Setup
```bash
# In another terminal, run the example
cd /home/shadeform/Tinkerbell
python examples/example_simple_concurrent.py

# Watch GPU usage during training/inference
watch -n 1 nvidia-smi
```

## 📚 Advanced: Manual Environment Setup

If you prefer to set environment variables manually:

```bash
# vLLM Configuration
export VLLM_MODEL="meta-llama/Llama-3.2-1B"
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTIL=0.4
export VLLM_ENABLE_LORA=true
export VLLM_MAX_LORAS=8
export VLLM_MAX_LORA_RANK=64
export VLLM_DTYPE=auto
export VLLM_TRUST_REMOTE_CODE=true

# Optional: GPU Selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: Megatron-Bridge Backend
export USE_MEGATRON_BRIDGE=true

# Start server
python src/app.py
```

## 🎓 Understanding GPU Distribution

### What happens with TP=4 (4 GPUs)?

```python
# Model: 1B parameters = ~2GB in fp16

# Single GPU:
GPU0: [Full Model 2GB] ← All parameters on one GPU

# TP=4 (4 GPUs):
GPU0: [Layer 0-5  : 500MB]  ← Shard 1
GPU1: [Layer 6-11 : 500MB]  ← Shard 2
GPU2: [Layer 12-17: 500MB]  ← Shard 3
GPU3: [Layer 18-23: 500MB]  ← Shard 4

# During inference:
# 1. Input goes to all GPUs
# 2. Each GPU processes its layers
# 3. Results combined at the end
# 4. All GPUs work in parallel!
```

### LoRA Adapters in Multi-GPU

LoRA adapters are small, so they're **replicated** on all GPUs:

```
Base Model (distributed):
GPU0: [Base Shard 1] + [LoRA User1, User2, ...]
GPU1: [Base Shard 2] + [LoRA User1, User2, ...]
GPU2: [Base Shard 3] + [LoRA User1, User2, ...]
GPU3: [Base Shard 4] + [LoRA User1, User2, ...]

Each GPU has:
- Its shard of base model (large, distributed)
- All LoRA adapters (small, replicated)
```

## 🚀 Recommended Configurations

### For Development/Testing
```bash
# 2 GPUs, low memory usage
./scripts/run_server_multigpu.sh HuggingFaceTB/SmolLM2-135M-Instruct 2 0.3
```

### For Production (Balanced)
```bash
# 4 GPUs, balanced memory
./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 4 0.4
```

### For Maximum Inference
```bash
# 4 GPUs, high vLLM memory
./scripts/run_server_multigpu.sh Qwen/Qwen3-0.6B 4 0.7
```

### For Maximum Training
```bash
# 2 GPUs, low vLLM memory (more for training)
export MAX_LORAS=4
./scripts/run_server_multigpu.sh HuggingFaceTB/SmolLM2-135M-Instruct 2 0.2
```

## 📝 Summary

**Your System**: 4 GPUs available

**Recommended Command**:
```bash
./scripts/run_server_4gpu.sh
```

This will:
- ✅ Use all 4 GPUs for vLLM tensor parallelism
- ✅ Allocate 40% memory per GPU for vLLM
- ✅ Leave 60% per GPU for Megatron-Bridge training
- ✅ Support up to 8 concurrent LoRA adapters
- ✅ Enable in-memory weight streaming

**Expected Performance**:
- ~3-4x faster inference vs single GPU
- Support 5-10 concurrent training users
- ~95GB total memory available

---

**Quick Reference**:
```bash
# Start with defaults (4 GPUs)
./scripts/run_server_4gpu.sh

# Start with 2 GPUs
./scripts/run_server_2gpu.sh

# Custom configuration
./scripts/run_server_multigpu.sh [model] [gpus] [memory]

# Check GPU status
nvidia-smi

# Stop server
Ctrl+C or: lsof -ti:8000 | xargs kill -9
```
