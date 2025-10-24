# Concurrent LoRA Training with Megatron Backend

## Overview

The Tink worker now supports **concurrent LoRA adapter training** with Megatron-LM backend using PEFT (Parameter-Efficient Fine-Tuning). Multiple users can create and train different LoRA adapters on the same base Megatron model.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Tink Worker (Megatron Mode)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐      ┌────────────────────────┐  │
│  │  Megatron Backend    │      │   vLLM Engine          │  │
│  │  (PEFT Multi-Adapter)│      │   (Co-located)         │  │
│  │                      │      │                        │  │
│  │  - Base Model (GPT)  │      │  - Fast Sampling       │  │
│  │  - LoRA Adapter 1    │──────▶  - LoRA Adapter 1     │  │
│  │  - LoRA Adapter 2    │──────▶  - LoRA Adapter 2     │  │
│  │  - LoRA Adapter N    │──────▶  - LoRA Adapter N     │  │
│  │                      │      │                        │  │
│  │  Training: Sequential│      │  Sampling: Concurrent  │  │
│  └──────────────────────┘      └────────────────────────┘  │
│                                                              │
│              Single Megatron GPTModel                        │
│              + PEFT Multi-Adapter Wrapper                    │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Multi-User Support**
- Multiple users can create different LoRA adapters
- Each adapter has a unique ID (e.g., `base_lora_abc123`)
- Adapters share the same base Megatron model (memory efficient)

### 2. **Sequential Training**
- Training is processed **sequentially** (one adapter at a time)
- This is a PEFT limitation: single `active_adapter` state
- Jobs are queued and processed in order
- No memory overhead from concurrent training

### 3. **Concurrent Sampling**
- vLLM handles sampling concurrently
- Each adapter can generate text simultaneously
- Fast inference with LoRA adapters

### 4. **Automatic vLLM Integration**
- When an adapter is created, it's automatically saved
- vLLM registers the adapter immediately
- Ready for sampling without manual steps

## Configuration

### Environment Variables

```bash
# Enable Megatron backend
export USE_MEGATRON=true

# Enable vLLM for fast sampling
export USE_VLLM=true
export VLLM_BASE_URL=http://localhost:8000
export VLLM_ENABLE_LORA=true
export VLLM_MAX_LORAS=4
export VLLM_GPU_MEMORY_UTIL=0.2  # Only 5% for vLLM

# Worker pool (for concurrent job submission)
export MAX_WORKERS=4
```

### Start Megatron Server

You'll need a Megatron server for sampling (or use vLLM instead):

```bash
# Terminal 1: Start Megatron server (optional if using vLLM)
# ... your Megatron server startup command ...

# Terminal 2: Start vLLM server (recommended)
trl vllm-serve \
    --model meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 4 \
    --port 8000

# Terminal 3: Start Tink worker with Megatron
torchrun --nproc_per_node=2 src/app.py
```

## How It Works

### 1. Adapter Creation

When a user calls `client.add_lora()`:

```python
# User: Alice
client.add_lora(base_model="base", rank=16, alpha=32)
# Returns: {"model_id": "base_lora_abc123"}
```

**Behind the scenes:**
1. `megatron_backend.create_lora_adapter()` is called
2. PEFT wraps the Megatron model (first adapter) or adds to existing wrapper (subsequent adapters)
3. Adapter is saved to `/tmp/lora_adapters/base_lora_abc123/`
4. vLLM registers the adapter automatically
5. Ready for both training and sampling

### 2. Training (Sequential)

When users call `client.fwdbwd()` + `client.optim()`:

```python
# User: Alice
client.fwdbwd(model_id="base_lora_abc123", data=[...])
client.optim(model_id="base_lora_abc123")

# User: Bob (concurrent submission, sequential processing)
client.fwdbwd(model_id="base_lora_def456", data=[...])
client.optim(model_id="base_lora_def456")
```

**Behind the scenes:**
1. Jobs are queued in `_work_queue`
2. `_training_lock` ensures sequential processing:
   - Alice's fwdbwd starts → switches to `base_lora_abc123` adapter
   - Alice's fwdbwd completes
   - Bob's fwdbwd starts → switches to `base_lora_def456` adapter
   - Bob's fwdbwd completes
3. Each adapter maintains its own:
   - Gradients (accumulated per adapter)
   - Optimizer state (separate Adam optimizer)
   - Training history

### 3. Sampling (Concurrent)

vLLM handles sampling concurrently for all adapters:

```python
# Alice and Bob can sample simultaneously
alice_result = client.sample(model_id="base_lora_abc123", prompts=["Hello"])
bob_result = client.sample(model_id="base_lora_def456", prompts=["Hello"])
```

**Behind the scenes:**
1. vLLM engine loads both adapters
2. Generates text concurrently
3. No interference between adapters

## Implementation Details

### PEFT Multi-Adapter Pattern

```python
# First adapter: Wrap the base model
peft_model = get_peft_model(base_model, lora_config)

# Subsequent adapters: Add to existing wrapper
peft_model.add_adapter("adapter_2", lora_config)
peft_model.add_adapter("adapter_3", lora_config)

# Switch between adapters during training
peft_model.set_adapter("adapter_1")  # Train adapter 1
# ... training loop ...

peft_model.set_adapter("adapter_2")  # Train adapter 2
# ... training loop ...
```

### Sequential Training Lock

```python
# Global lock ensures only one adapter trains at a time
with _training_lock:
    peft_model.set_adapter(model_id)
    # ... forward-backward pass ...
```

### Per-Adapter State

Each adapter maintains:
- **Gradients**: Accumulated during forward-backward
- **Optimizer**: Separate Adam optimizer with independent state
- **Metadata**: Created time, rank, alpha, etc.

## Memory Efficiency

### vs. Separate Model Copies

**Without PEFT Multi-Adapter (naive approach):**
```
Base Model:   7B params × 4 bytes = 28 GB
Adapter 1:    28 GB (full model copy)
Adapter 2:    28 GB (full model copy)
Adapter 3:    28 GB (full model copy)
Total:        112 GB (OOM!)
```

**With PEFT Multi-Adapter:**
```
Base Model:   7B params × 4 bytes = 28 GB
Adapter 1:    ~0.5% × 28 GB     = 140 MB (LoRA params only)
Adapter 2:    ~0.5% × 28 GB     = 140 MB
Adapter 3:    ~0.5% × 28 GB     = 140 MB
Total:        ~28.4 GB (fits easily!)
```

### vLLM Memory

vLLM uses only 5% GPU memory by default:
```
vLLM:         5% × GPU memory   = ~600 MB (on 12GB GPU)
Training:     95% × GPU memory  = ~11.4 GB
```

## Example: Concurrent Users

```python
# Three users training simultaneously
import concurrent.futures
from tinker import Client

def train_user(user_name):
    client = Client("http://localhost:8000")

    # Create adapter
    adapter = client.add_lora(base_model="base", rank=16, alpha=32)
    model_id = adapter["model_id"]
    print(f"[{user_name}] Created: {model_id}")

    # Training loop
    for step in range(3):
        # Forward-backward
        result = client.fwdbwd(
            model_id=model_id,
            data=[{"messages": [
                {"role": "user", "content": f"Hello from {user_name}!"},
                {"role": "assistant", "content": f"Hi {user_name}!"}
            ]}]
        )
        print(f"[{user_name}] Step {step+1}, Loss: {result['loss']:.4f}")

        # Optimizer step
        client.optim(model_id=model_id, learning_rate=1e-4)

    # Test sampling
    sample = client.sample(model_id=model_id, prompts=["Hello"])
    print(f"[{user_name}] Generated: {sample['sequences'][0]['tokens']}")

# Run 3 users concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(train_user, "Alice"),
        executor.submit(train_user, "Bob"),
        executor.submit(train_user, "Charlie")
    ]
    concurrent.futures.wait(futures)
```

**Output:**
```
[Alice] Created: base_lora_abc123
[Bob] Created: base_lora_def456
[Charlie] Created: base_lora_ghi789

[Alice] Switched to adapter, starting forward-backward pass
[Alice] Step 1, Loss: 2.3456
[Alice] Optimizer step complete. LR: 0.0001

[Bob] Switched to adapter, starting forward-backward pass
[Bob] Step 1, Loss: 2.4567
[Bob] Optimizer step complete. LR: 0.0001

[Charlie] Switched to adapter, starting forward-backward pass
[Charlie] Step 1, Loss: 2.5678
[Charlie] Optimizer step complete. LR: 0.0001

... (continues sequentially) ...

[Alice] Generated: [128, 256, 512, ...]
[Bob] Generated: [128, 257, 513, ...]
[Charlie] Generated: [128, 258, 514, ...]
```

## Comparison: Concurrent vs Sequential

| Aspect | Concurrent (HF) | Sequential (Megatron+PEFT) |
|--------|----------------|---------------------------|
| **Adapter Creation** | Concurrent | Concurrent |
| **Training** | Concurrent (OOM risk) | Sequential (memory safe) |
| **Sampling** | N/A (slow) | Concurrent (vLLM) |
| **Memory Usage** | High (copies) | Low (shared base) |
| **Throughput** | High (if memory permits) | Medium (queued) |
| **Best For** | Small models, few users | Large models, many users |

## Troubleshooting

### Error: "Base Megatron model not initialized"

**Cause:** `megatron_backend.initialize_base_model()` wasn't called

**Solution:** Ensure Megatron backend is initialized in worker.py:
```python
if USE_MEGATRON:
    model, optimizer, data_iterator = _initialize_megatron()
    megatron_backend.initialize_base_model(model, model.config, optimizer)
```

### Error: "Adapter not found"

**Cause:** Trying to train/sample with non-existent adapter

**Solution:** Create adapter first with `client.add_lora()`

### Slow Training

**Cause:** Sequential processing means jobs are queued

**Solution:** This is expected behavior. Consider:
- Batch multiple training steps together
- Use larger batch sizes
- Reduce number of training steps per user

## Summary

The Megatron backend now supports:

1. ✅ **Concurrent adapter creation** - Multiple users can create adapters simultaneously
2. ✅ **Sequential training** - Training processes one adapter at a time (PEFT limitation)
3. ✅ **Concurrent sampling** - vLLM handles inference for all adapters in parallel
4. ✅ **Memory efficient** - Shared base model, only LoRA params duplicated
5. ✅ **Automatic vLLM sync** - Adapters auto-registered after creation

**Key Insight:** Sequential training is NOT a limitation for multi-user scenarios because:
- Adapter creation is instant (concurrent)
- Training jobs are short (few seconds each)
- Queue processing is fast
- Sampling remains concurrent via vLLM

This architecture provides the best balance of memory efficiency, concurrent user support, and fast inference!
