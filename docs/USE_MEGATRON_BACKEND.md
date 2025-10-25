# Using Megatron-Bridge Backend

## Quick Answer

To use **Megatron-Bridge** instead of HuggingFace backend:

### Option 1: Use the Multi-GPU Script (Easiest)
```bash
cd /home/shadeform/Tinkerbell
export TRAINING_BACKEND=megatron-bridge
./scripts/run_server_4gpu.sh
```

The script automatically sets `USE_MEGATRON=true` when you set `TRAINING_BACKEND=megatron-bridge`.

### Option 2: Set Environment Variable Manually
```bash
export USE_MEGATRON=true
export USE_VLLM=true
python3.12 src/app.py
```

### Option 3: Source the Config Script
```bash
source scripts/set_megatron_backend.sh
python3.12 src/app.py
```

---

## What Changed?

### Before (HuggingFace Backend)
```
Worker pool initialized. Active backends: HuggingFace (training/sampling)
```

### After (Megatron-Bridge Backend)
```
Worker pool initialized. Active backends: Megatron-Bridge (training), vLLM (sampling)
```

---

## Complete Setup

### Step 1: Set Environment Variables
```bash
# Enable Megatron-Bridge for training
export USE_MEGATRON=true

# Enable vLLM for inference
export USE_VLLM=true

# Configure vLLM multi-GPU
export VLLM_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTIL=0.4

# Optional: Configure Megatron
export MEGATRON_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
export MEGATRON_TP_SIZE=1
export MEGATRON_PP_SIZE=1
```

### Step 2: Start Server
```bash
python3.12 src/app.py
```

You should see:
```
✓ Megatron-Bridge backend initialized
✓ vLLM engine initialized
Worker pool initialized. Active backends: Megatron-Bridge (training), vLLM (sampling)
```

---

## Verification

### Check Active Backend

```bash
# Server logs will show:
# "Worker pool initialized. Active backends: Megatron-Bridge..."

# Or via API:
curl http://localhost:8000/api/v1/backend_info
```

### Test Training with Megatron-Bridge

```python
import requests

# Create LoRA adapter (uses Megatron-Bridge)
response = requests.post("http://localhost:8000/api/v1/add_lora", json={
    "base_model": "base",
    "rank": 16,
    "alpha": 32
})
model_id = response.json()["model_id"]

# Train (uses Megatron-Bridge)
training_data = [[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]]

response = requests.post("http://localhost:8000/api/v1/fwdbwd", json={
    "model_id": model_id,
    "data": training_data,
    "loss_fn": "cross_entropy"
})

print(f"Loss: {response.json()['loss']}")
# Backend info will show "megatron-bridge"
```

---

## Backend Selection Logic

From `worker.py` line 78-94:

```python
USE_MEGATRON = MEGATRON_AVAILABLE and os.environ.get("USE_MEGATRON", "false").lower() == "true"
USE_VLLM = VLLM_BACKEND_AVAILABLE and os.environ.get("USE_VLLM", "false").lower() == "true"

def get_backend() -> str:
    """Get the current backend being used."""
    if USE_MEGATRON:
        return "megatron"
    elif USE_VLLM:
        return "vllm"
    else:
        return "huggingface"
```

**Priority:**
1. Megatron-Bridge (if `USE_MEGATRON=true`)
2. vLLM (if `USE_VLLM=true`)
3. HuggingFace (default)

---

## Configuration Matrix

| Environment Variable | Value | Effect |
|---------------------|-------|--------|
| `USE_MEGATRON` | `true` | Uses Megatron-Bridge for training |
| `USE_MEGATRON` | `false` | Uses HuggingFace for training (default) |
| `USE_VLLM` | `true` | Uses vLLM for inference |
| `USE_VLLM` | `false` | Uses HuggingFace for inference |

### Common Configurations

#### 1. Megatron-Bridge + vLLM (Recommended)
```bash
export USE_MEGATRON=true
export USE_VLLM=true
# Training: Megatron-Bridge (fast, scalable)
# Inference: vLLM (fast, multi-GPU)
```

#### 2. HuggingFace Only (Simplest)
```bash
# Don't set any variables
# Training: HuggingFace (simple, slow)
# Inference: HuggingFace (simple, slow)
```

#### 3. Megatron-Bridge + HuggingFace
```bash
export USE_MEGATRON=true
export USE_VLLM=false
# Training: Megatron-Bridge (fast)
# Inference: HuggingFace (slower, but no vLLM needed)
```

---

## Example Commands

### 1. Start with Megatron-Bridge (4 GPUs)
```bash
cd /home/shadeform/Tinkerbell
export TRAINING_BACKEND=megatron-bridge
./scripts/run_server_4gpu.sh
```

### 2. Start with Megatron-Bridge (Manual)
```bash
cd /home/shadeform/Tinkerbell

export USE_MEGATRON=true
export USE_VLLM=true
export VLLM_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTIL=0.4

python3.12 src/app.py
```

### 3. Use Helper Script
```bash
cd /home/shadeform/Tinkerbell
source scripts/set_megatron_backend.sh
python3.12 src/app.py
```

---

## Troubleshooting

### Issue: Still shows "HuggingFace" backend

**Check 1:** Environment variable set?
```bash
echo $USE_MEGATRON
# Should print: true
```

**Check 2:** Megatron-Bridge installed?
```bash
python3.12 -c "from megatron.bridge import AutoBridge; print('✓ Bridge OK')"
```

**Check 3:** Server restarted?
```bash
# Kill and restart server
lsof -ti:8000 | xargs kill -9
export USE_MEGATRON=true
export USE_VLLM=true
python3.12 src/app.py
```

### Issue: "Megatron-Bridge not available"

**Solution:** Install Megatron-Bridge
```bash
cd /home/shadeform/refs/Megatron-Bridge
pip install -e .

# Verify
python3.12 -c "from megatron.bridge import AutoBridge; print('Installed!')"
```

### Issue: "MEGATRON_AVAILABLE = False"

This means the old Megatron-LM import failed. This is OK! The new backend uses Megatron-Bridge, which imports differently.

**Check worker.py line 14-36** - it tries to import old Megatron-LM
**But megatron_backend.py line 21-33** - uses new Megatron-Bridge

The worker check is outdated. Update it:

```python
# worker.py line 14-36: OLD check for Megatron-LM
# megatron_backend.py line 21-33: NEW check for Megatron-Bridge

# As long as megatron_backend.py imports successfully, you're good!
```

---

## Verify Backend is Working

### Method 1: Check Server Logs
```
Look for:
✓ "Megatron-Bridge backend initialized"
✓ "Worker pool initialized. Active backends: Megatron..."
```

### Method 2: Check Backend Info Endpoint
```bash
curl http://localhost:8000/api/v1/backend_info | python3.12 -m json.tool
```

Should return:
```json
{
  "initialized": true,
  "backend": "megatron-bridge",
  "bridge_available": true,
  "model_name": "...",
  "active_adapters": 0
}
```

### Method 3: Run Example
```bash
python3.12 examples/example_simple_concurrent.py
```

Check output for:
```
[Alice] Created adapter with Megatron-Bridge backend
[Bob] Training with Megatron-Bridge...
```

---

## Quick Reference Card

```bash
# ═══════════════════════════════════════════
#  Use Megatron-Bridge Backend
# ═══════════════════════════════════════════

# EASY WAY (Recommended)
export TRAINING_BACKEND=megatron-bridge
./scripts/run_server_4gpu.sh

# MANUAL WAY
export USE_MEGATRON=true
export USE_VLLM=true
python3.12 src/app.py

# HELPER SCRIPT WAY
source scripts/set_megatron_backend.sh
python3.12 src/app.py

# ═══════════════════════════════════════════
#  Verify It's Working
# ═══════════════════════════════════════════

# Check logs
# Should see: "Active backends: Megatron-Bridge"

# Check environment
echo $USE_MEGATRON
echo $USE_VLLM

# Test it
python3.12 examples/example_simple_concurrent.py
```

---

## Summary

**To use Megatron-Bridge backend, just:**

```bash
export TRAINING_BACKEND=megatron-bridge
./scripts/run_server_4gpu.sh
```

**That's it!** The script handles everything:
- ✅ Sets `USE_MEGATRON=true`
- ✅ Sets `USE_VLLM=true`
- ✅ Configures multi-GPU vLLM
- ✅ Starts server with both backends

**You'll see:**
```
Worker pool initialized. Active backends: Megatron-Bridge (training), vLLM (sampling)
```

Instead of:
```
Worker pool initialized. Active backends: HuggingFace (training/sampling)
```
