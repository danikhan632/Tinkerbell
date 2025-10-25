# Tinkerbell Server Launch Scripts

## Quick Launch Scripts

### 🟢 run_server_4gpu.sh (Recommended)
**Uses all 4 GPUs for maximum performance**

```bash
./scripts/run_server_4gpu.sh
```

**Configuration:**
- Model: SmolLM2-135M-Instruct
- GPUs: 4 (tensor parallelism)
- Memory: 40% per GPU for vLLM, 60% for training
- Expected: ~160 tokens/sec inference

---

### 🟡 run_server_2gpu.sh
**Uses 2 GPUs for balanced performance**

```bash
./scripts/run_server_2gpu.sh
```

**Configuration:**
- Model: SmolLM2-135M-Instruct
- GPUs: 2 (tensor parallelism)
- Memory: 40% per GPU for vLLM, 60% for training
- Expected: ~90 tokens/sec inference

---

### 🔵 run_server_multigpu.sh (Custom)
**Flexible configuration for any setup**

```bash
./scripts/run_server_multigpu.sh [MODEL] [NUM_GPUS] [GPU_MEMORY_UTIL]
```

**Examples:**

```bash
# Llama 3.2-1B on all 4 GPUs
./scripts/run_server_multigpu.sh meta-llama/Llama-3.2-1B 4 0.4

# Qwen3 on 2 GPUs with high memory
./scripts/run_server_multigpu.sh Qwen/Qwen3-0.6B 2 0.6

# SmolLM on 4 GPUs with low vLLM memory (more for training)
./scripts/run_server_multigpu.sh HuggingFaceTB/SmolLM2-135M-Instruct 4 0.3

# Large model across all GPUs
./scripts/run_server_multigpu.sh meta-llama/Llama-3.1-8B 4 0.5
```

---

## Parameters

| Parameter | Default | Description | Example Values |
|-----------|---------|-------------|----------------|
| `MODEL` | SmolLM2-135M-Instruct | HuggingFace model | `meta-llama/Llama-3.2-1B`, `Qwen/Qwen3-0.6B` |
| `NUM_GPUS` | 4 | Number of GPUs (tensor parallel size) | `1`, `2`, `4` |
| `GPU_MEMORY_UTIL` | 0.4 | Fraction of GPU memory for vLLM | `0.2` (training heavy), `0.7` (inference heavy) |

---

## Environment Variables

Set these **before** running the scripts for additional control:

```bash
# LoRA Configuration
export MAX_LORAS=8              # Max concurrent LoRA adapters (default: 8)
export MAX_LORA_RANK=64         # Max LoRA rank (default: 64)

# Training Backend
export TRAINING_BACKEND="megatron-bridge"  # or "huggingface"

# GPU Selection (use specific GPUs)
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Then run any script
./scripts/run_server_4gpu.sh
```

---

## Visual Guide

```
┌─────────────────────────────────────────────────────┐
│              Your System: 4 GPUs                    │
└─────────────────────────────────────────────────────┘

Option 1: ALL 4 GPUs (Fastest)
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ GPU0 │ │ GPU1 │ │ GPU2 │ │ GPU3 │
│ 40%  │ │ 40%  │ │ 40%  │ │ 40%  │  ← vLLM
│ 60%  │ │ 60%  │ │ 60%  │ │ 60%  │  ← Training
└──────┘ └──────┘ └──────┘ └──────┘
Command: ./scripts/run_server_4gpu.sh
Speed:   ~160 tokens/sec


Option 2: 2 GPUs (Balanced)
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ GPU0 │ │ GPU1 │ │      │ │      │
│ 40%  │ │ 40%  │ │ FREE │ │ FREE │  ← vLLM
│ 60%  │ │ 60%  │ │      │ │      │  ← Training
└──────┘ └──────┘ └──────┘ └──────┘
Command: ./scripts/run_server_2gpu.sh
Speed:   ~90 tokens/sec


Option 3: Custom
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ GPU0 │ │ GPU1 │ │ GPU2 │ │ GPU3 │
│  X%  │ │  X%  │ │  X%  │ │  X%  │  ← vLLM (you choose)
│(1-X%)│ │(1-X%)│ │(1-X%)│ │(1-X%)│  ← Training
└──────┘ └──────┘ └──────┘ └──────┘
Command: ./scripts/run_server_multigpu.sh MODEL GPUS MEM
Example: ./scripts/run_server_multigpu.sh Qwen/Qwen3-0.6B 4 0.5
```

---

## Memory Allocation Examples

### Conservative (Training Heavy)
```bash
./scripts/run_server_multigpu.sh YourModel 4 0.3
```
- vLLM: 30% × 4 GPUs = ~29 GB total
- Training: 70% × 4 GPUs = ~67 GB total
- Use case: Many concurrent training users

### Balanced (Default)
```bash
./scripts/run_server_4gpu.sh
```
- vLLM: 40% × 4 GPUs = ~38 GB total
- Training: 60% × 4 GPUs = ~58 GB total
- Use case: Mixed workload

### Aggressive (Inference Heavy)
```bash
./scripts/run_server_multigpu.sh YourModel 4 0.7
```
- vLLM: 70% × 4 GPUs = ~67 GB total
- Training: 30% × 4 GPUs = ~29 GB total
- Use case: High concurrent inference requests

---

## Monitoring

### Watch GPU Usage in Real-Time
```bash
watch -n 1 nvidia-smi
```

### Check Specific GPU Stats
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1
```

### Check Server Port
```bash
lsof -i:8000
```

---

## Common Commands

```bash
# Start server (4 GPUs)
cd /home/shadeform/Tinkerbell
./scripts/run_server_4gpu.sh

# In another terminal: test it
cd /home/shadeform/Tinkerbell
python3.12 examples/example_simple_concurrent.py

# Monitor GPUs
watch -n 1 nvidia-smi

# Stop server
# Press Ctrl+C or:
lsof -ti:8000 | xargs kill -9
```

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i:8000

# Kill existing server
lsof -ti:8000 | xargs kill -9

# Check GPU availability
nvidia-smi

# Check vLLM installation
python3.12 -c "import vllm; print(vllm.__version__)"
```

### Out of memory errors
```bash
# Reduce memory allocation
./scripts/run_server_multigpu.sh YourModel 4 0.3  # Use 30% instead of 40%

# Or use fewer GPUs
./scripts/run_server_multigpu.sh YourModel 2 0.4

# Or reduce max LoRAs
export MAX_LORAS=4
./scripts/run_server_4gpu.sh
```

### Slow performance
```bash
# Check GPU communication
nvidia-smi topo -m  # Should show NVLink between GPUs

# Check GPU utilization
nvidia-smi  # All GPUs should be ~80-100% utilized during inference
```

---

## Architecture

```
┌────────────────────────────────────────┐
│    Tinkerbell Server (Port 8000)      │
│                                        │
│  ┌──────────────┐  ┌───────────────┐  │
│  │     vLLM     │  │   Megatron-   │  │
│  │   (Inference)│◄─┤    Bridge     │  │
│  │    4 GPUs    │  │  (Training)   │  │
│  └──────────────┘  └───────────────┘  │
│         │                  │           │
│         └─── Weight ───────┘           │
│            Streaming                   │
│          (In-Memory!)                  │
└────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  GPU 0  GPU 1 │
    │  GPU 2  GPU 3 │
    └───────────────┘
```

---

## More Information

- **Full Setup Guide**: `../MULTIGPU_SETUP.md`
- **Quick Start**: `../QUICK_START.md`
- **Migration Guide**: `../MEGATRON_BRIDGE_MIGRATION.md`
- **Changes Summary**: `../CHANGES_SUMMARY.md`

---

**Ready to start?**

```bash
cd /home/shadeform/Tinkerbell
./scripts/run_server_4gpu.sh
```
