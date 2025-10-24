# Flask Tinker Server - Concurrent Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Client Applications                          │
│  (Multiple concurrent users training different LoRA adapters)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP Requests
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Flask App (app.py)                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Endpoints: /fwdbwd, /optim_step, /add_lora, /sample, etc.       │ │
│  │  - Creates future_id for each request                             │ │
│  │  - Updates futures_store (thread-safe with lock)                  │ │
│  │  - Enqueues jobs to work_queue                                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ enqueue_job()
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Work Queue (queue.Queue)                             │
│  FIFO queue containing: (job_type, request_id, params_json)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ _work_queue.get()
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Worker Dispatcher Thread (_worker_loop)                   │
│  - Runs in background daemon thread                                    │
│  - Initializes base model (one-time, thread-safe)                      │
│  - Dequeues jobs from work_queue                                       │
│  - Submits jobs to ThreadPoolExecutor                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ thread_pool.submit()
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ThreadPoolExecutor (MAX_WORKERS threads)                   │
│                                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │
│  │ Worker Thread 1│  │ Worker Thread 2│  │ Worker Thread 3│  ...      │
│  │                │  │                │  │                │           │
│  │ User A:        │  │ User B:        │  │ User C:        │           │
│  │ LoRA 1 training│  │ LoRA 2 training│  │ LoRA 3 inference│          │
│  │                │  │                │  │                │           │
│  │ Calls:         │  │ Calls:         │  │ Calls:         │           │
│  │ _process_job() │  │ _process_job() │  │ _process_job() │           │
│  └────────────────┘  └────────────────┘  └────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ hf_backend function calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  HuggingFace Backend (hf_backend.py)                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    Global Shared State                         │   │
│  │  - base_model (frozen, shared by all adapters)                 │   │
│  │  - tokenizer (shared, read-only)                               │   │
│  │  - lora_adapters = {"adapter_1": PeftModel, "adapter_2": ...}  │   │
│  │  - optimizers = {"adapter_1": Adam, "adapter_2": ...}          │   │
│  │  - gradients_accumulated = {"adapter_1": bool, ...}            │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    Thread Safety Locks                         │   │
│  │  - _base_model_lock (RLock) → Protects base model init        │   │
│  │  - _adapters_lock (RLock) → Protects adapter dict ops         │   │
│  │  - _adapter_locks[model_id] (RLock) → Per-adapter operations  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              Thread-Safe Operations                            │   │
│  │                                                                 │   │
│  │  create_lora_adapter(model_id):                                │   │
│  │    with _adapters_lock:                                        │   │
│  │      - Check if exists, create if not                          │   │
│  │      - Add to lora_adapters dict                               │   │
│  │                                                                 │   │
│  │  forward_backward(model_id):                                   │   │
│  │    with _get_adapter_lock(model_id):  # Per-adapter lock       │   │
│  │      - Run forward pass                                        │   │
│  │      - Compute loss                                            │   │
│  │      - Run backward pass                                       │   │
│  │      - Accumulate gradients                                    │   │
│  │                                                                 │   │
│  │  optim_step(model_id):                                         │   │
│  │    with _get_adapter_lock(model_id):  # Per-adapter lock       │   │
│  │      - Apply gradients to adapter weights                      │   │
│  │      - Zero gradients                                          │   │
│  │                                                                 │   │
│  │  generate(model_id):                                           │   │
│  │    with _get_adapter_lock(model_id):  # Per-adapter lock       │   │
│  │      - Set model to eval mode                                  │   │
│  │      - Run inference                                           │   │
│  │      - Return generated tokens                                 │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ PyTorch operations
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            GPU / CUDA                                   │
│  - Base model weights (frozen, shared)                                 │
│  - LoRA adapter weights (trainable, per-adapter)                       │
│  - Optimizer states (per-adapter)                                      │
│  - Activations and gradients (per-request)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Concurrency Flow Example

### Scenario: 3 Users Training Different LoRAs Concurrently

```
Time →

User A: POST /fwdbwd {model_id: "lora_1"}
        │
        ├──→ enqueue_job("fwdbwd", "req_1", {...})
        │
        └──→ Worker Thread 1 (acquires lock for lora_1)
             │
             ├──→ forward_backward(lora_1)
             │    ┌─────────────────────────────────┐
             │    │ with _get_adapter_lock("lora_1"): │
             │    │   - Forward pass               │
             │    │   - Backward pass              │
             │    │   - Gradient accumulation      │
             │    └─────────────────────────────────┘
             │
             └──→ Update futures_store[req_1] = {status: "completed", result: {...}}

User B: POST /fwdbwd {model_id: "lora_2"}  (runs concurrently with User A)
        │
        ├──→ enqueue_job("fwdbwd", "req_2", {...})
        │
        └──→ Worker Thread 2 (acquires lock for lora_2)
             │
             ├──→ forward_backward(lora_2)  ← Different lock, no blocking!
             │    ┌─────────────────────────────────┐
             │    │ with _get_adapter_lock("lora_2"): │
             │    │   - Forward pass               │
             │    │   - Backward pass              │
             │    │   - Gradient accumulation      │
             │    └─────────────────────────────────┘
             │
             └──→ Update futures_store[req_2] = {status: "completed", result: {...}}

User C: POST /sample {model_id: "lora_3"}  (runs concurrently with A & B)
        │
        ├──→ enqueue_job("sample", "req_3", {...})
        │
        └──→ Worker Thread 3 (acquires lock for lora_3)
             │
             ├──→ generate(lora_3)  ← Different lock, no blocking!
             │    ┌─────────────────────────────────┐
             │    │ with _get_adapter_lock("lora_3"): │
             │    │   - Inference                  │
             │    │   - Token generation           │
             │    └─────────────────────────────────┘
             │
             └──→ Update futures_store[req_3] = {status: "completed", result: {...}}

All three operations run in parallel! ✓
```

## Locking Strategy

### Three-Level Locking Hierarchy

```
1. Base Model Lock (_base_model_lock)
   │
   │  Purpose: Protect one-time initialization
   │  Scope: Entire base model
   │  Usage: initialize_base_model()
   │
   └──→ Held briefly during initialization only

2. Adapters Dictionary Lock (_adapters_lock)
   │
   │  Purpose: Protect adapter dictionary operations
   │  Scope: lora_adapters, optimizers, gradients_accumulated dicts
   │  Usage: create_lora_adapter(), delete_adapter(), adapter lookups
   │
   └──→ Held briefly during dict modifications only

3. Per-Adapter Locks (_adapter_locks[model_id])
   │
   │  Purpose: Serialize operations on the SAME adapter
   │  Scope: Single adapter's training/inference operations
   │  Usage: forward_backward(), optim_step(), generate()
   │
   └──→ Held during entire operation (training step, inference)
      ├──→ Allows concurrent ops on different adapters ✓
      └──→ Prevents race conditions on same adapter ✓
```

### Lock Acquisition Order (prevents deadlocks)

```
Always acquire locks in this order:
1. _adapters_lock (if needed)
2. _adapter_locks[model_id] (if needed)
3. _base_model_lock (rare, only during init)

Never:
- Acquire multiple adapter locks simultaneously
- Hold locks while waiting for I/O
- Hold locks while sleeping
```

## Thread Safety Guarantees

### ✅ Safe Concurrent Operations

```python
# These can run concurrently without blocking:

Thread 1: forward_backward(model_id="lora_1", ...)
Thread 2: forward_backward(model_id="lora_2", ...)
Thread 3: generate(model_id="lora_3", ...)
Thread 4: optim_step(model_id="lora_4", ...)
```

### ⚠️ Serialized Operations (by design)

```python
# These will wait for each other (same adapter):

Thread 1: forward_backward(model_id="lora_1", ...)  # Acquires lock
Thread 2: forward_backward(model_id="lora_1", ...)  # Waits for Thread 1
Thread 3: generate(model_id="lora_1", ...)          # Waits for Threads 1 & 2
```

## Resource Sharing Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Resources                         │
│  - Base model weights (frozen, read-only) → NO LOCKING     │
│  - Tokenizer (read-only) → NO LOCKING                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Per-Adapter Resources                      │
│  - LoRA weights → LOCKED (per adapter)                     │
│  - Optimizer state → LOCKED (per adapter)                  │
│  - Gradient accumulation flag → LOCKED (per adapter)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Per-Request Resources                      │
│  - Activations → NO LOCKING (isolated to thread)           │
│  - Gradients → NO LOCKING (isolated to thread)             │
│  - DataLoader → NO LOCKING (created per request)           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Parameters

### Worker Pool Size (`MAX_WORKERS`)

```
MAX_WORKERS = 1:  Sequential processing (no concurrency)
MAX_WORKERS = 2:  Up to 2 adapters can train concurrently
MAX_WORKERS = 4:  Up to 4 adapters can train concurrently (default)
MAX_WORKERS = 8:  Up to 8 adapters can train concurrently (high concurrency)

Trade-offs:
- Higher = More concurrency, but higher GPU memory usage
- Lower = Less concurrency, but lower GPU memory usage

Recommended: Start with 4, tune based on GPU memory
```

### GPU Memory Considerations

```
Per-Adapter Memory Usage:
- LoRA weights: ~1-10 MB (rank dependent)
- Optimizer state: ~2-20 MB (rank dependent)
- Activations (during training): ~100-500 MB (batch size dependent)

Example (RTX 3090 with 24GB VRAM):
- Base model: ~500 MB
- 4 concurrent adapters (r=16): ~2 GB
- Remaining for activations: ~21 GB
→ MAX_WORKERS=4 is safe
```

## Monitoring and Debugging

### Key Metrics to Monitor

```bash
# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Thread pool activity (check logs)
Worker pool size: 4 threads
Worker pool initialized. Concurrent job processing: Enabled (HF)

# Per-adapter operations (check logs)
Created LoRA adapter 'base_lora_abc123'
[base_lora_abc123] Forward-backward complete. Loss: 1.2345, Samples: 1
[base_lora_abc123] Optimizer step complete. LR: 0.001
```

### Debug Checklist

```
Issue: Deadlock or hanging
→ Check lock acquisition order
→ Check for exceptions in worker threads
→ Verify no infinite waits

Issue: GPU OOM
→ Reduce MAX_WORKERS
→ Use smaller LoRA ranks
→ Reduce batch sizes

Issue: Slow despite concurrency
→ Verify users use different adapter IDs
→ Check if all requests target same adapter (serialized)
→ Monitor GPU utilization (should be high)
```

## Summary

This architecture enables:
- ✅ **True concurrent multi-user support**
- ✅ **Thread-safe adapter management**
- ✅ **Efficient GPU utilization**
- ✅ **Scalable to many users (limited by GPU memory)**
- ✅ **No race conditions or data corruption**
- ✅ **Clear concurrency semantics**

The key insight: **Per-adapter locking** allows different users to work on different adapters concurrently while preventing conflicts when multiple requests target the same adapter.
