# Concurrent Multi-User Support

This document describes the concurrency improvements made to the Flask Tinker server to support multiple concurrent users with different LoRA adapters.

## Overview

The server now supports **true concurrent processing** of multiple users training different LoRA adapters simultaneously. This is achieved through:

1. **Thread-safe backend operations** with per-adapter locking
2. **Thread pool for concurrent job processing** (HuggingFace backend)
3. **Thread-safe shared state management**

## Architecture

### Thread Safety Model

#### Per-Adapter Locking
- Each LoRA adapter has its own lock (`_adapter_locks`)
- Multiple users can train **different adapters** concurrently
- Operations on the **same adapter** are serialized to ensure consistency
- Base model has a separate lock for initialization

#### Shared Resource Locking
- `_base_model_lock`: Protects base model initialization (one-time operation)
- `_adapters_lock`: Protects adapter dictionary operations (create/delete)
- `_futures_store_lock`: Protects job status updates in the futures store

### Worker Pool Architecture

#### HuggingFace Backend (Concurrent)
```
┌─────────────────────────────────────────────────────────┐
│                    Flask App (Main Thread)              │
│  - Receives HTTP requests                               │
│  - Enqueues jobs to work queue                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Worker Dispatcher (Single Thread)          │
│  - Dequeues jobs from work queue                        │
│  - Submits jobs to thread pool                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Thread Pool (MAX_WORKERS threads)              │
│                                                          │
│  Thread 1: User A - LoRA 1 training                     │
│  Thread 2: User B - LoRA 2 training                     │
│  Thread 3: User C - LoRA 3 inference                    │
│  Thread 4: User D - LoRA 4 training                     │
└─────────────────────────────────────────────────────────┘
```

#### Megatron Backend (Sequential)
The Megatron backend does not yet support concurrent processing due to model state limitations. Jobs are processed sequentially.

## Configuration

### Environment Variables

- `MAX_WORKERS`: Number of worker threads for concurrent processing (default: 4)
  - Recommended: 2-8 threads depending on GPU memory and CPU cores
  - Higher values allow more concurrent users but increase memory usage

- `USE_MEGATRON`: Enable Megatron backend (default: false)
  - When enabled, concurrent processing is disabled
  - All jobs are processed sequentially

### Example Configuration

```bash
# For maximum concurrency with HuggingFace backend
export MAX_WORKERS=8

# Start the server
python app.py
```

## Usage Scenarios

### Scenario 1: Multiple Users Training Different Adapters
✅ **Fully Concurrent** - Each user's training runs in parallel

```python
# User A trains adapter_1
POST /fwdbwd {"model_id": "adapter_1", ...}

# User B trains adapter_2 (runs concurrently with User A)
POST /fwdbwd {"model_id": "adapter_2", ...}

# User C trains adapter_3 (runs concurrently with A and B)
POST /fwdbwd {"model_id": "adapter_3", ...}
```

### Scenario 2: Multiple Users Using Same Adapter
⚠️ **Serialized** - Operations are queued to prevent race conditions

```python
# User A trains adapter_1
POST /fwdbwd {"model_id": "adapter_1", ...}

# User B trains adapter_1 (waits for User A to finish)
POST /fwdbwd {"model_id": "adapter_1", ...}
```

### Scenario 3: Mixed Operations
✅ **Partially Concurrent**

```python
# User A trains adapter_1
POST /fwdbwd {"model_id": "adapter_1", ...}

# User B runs inference on adapter_1 (waits for training)
POST /api/v1/sample {"model_id": "adapter_1", ...}

# User C trains adapter_2 (runs concurrently with User A)
POST /fwdbwd {"model_id": "adapter_2", ...}
```

## Performance Benefits

### Concurrent Processing
- **Multiple adapters trained simultaneously**: Up to MAX_WORKERS adapters can train in parallel
- **Better resource utilization**: GPU and CPU cores are utilized more efficiently
- **Lower latency for independent operations**: Users don't wait for unrelated jobs

### Benchmark Results
Using the test script with 4 concurrent users:

```bash
python test_concurrent_users.py 4
```

Expected speedup: **2-4x** compared to sequential processing (depending on hardware)

## Testing

### Test Script
A comprehensive test script is provided: `test_concurrent_users.py`

```bash
# Test with 4 concurrent users (default)
python test_concurrent_users.py

# Test with 8 concurrent users
python test_concurrent_users.py 8
```

The test script simulates:
1. Creating unique LoRA adapters for each user
2. Training each adapter with multiple forward-backward passes
3. Running optimizer steps
4. Testing inference
5. Measuring total time and speedup

### Expected Output
```
================================================================================
Testing 4 concurrent users with different LoRA adapters
================================================================================

✓ Server is healthy

[User user_0] Creating LoRA adapter...
[User user_1] Creating LoRA adapter...
[User user_2] Creating LoRA adapter...
[User user_3] Creating LoRA adapter...

... (training logs) ...

================================================================================
RESULTS SUMMARY
================================================================================

Total users: 4
Successful: 4
Failed: 0
Total elapsed time: 12.34s
Average time per user: 11.52s

Per-user details:
  ✓ user_0: 11.23s (model: base_lora_abc123)
  ✓ user_1: 11.45s (model: base_lora_def456)
  ✓ user_2: 11.67s (model: base_lora_ghi789)
  ✓ user_3: 11.72s (model: base_lora_jkl012)

Concurrency speedup: 3.74x
(Estimated sequential time: 46.07s vs actual: 12.34s)

================================================================================
```

## Thread Safety Guarantees

### What is Thread-Safe
✅ Creating adapters concurrently
✅ Training different adapters concurrently
✅ Inference on different adapters concurrently
✅ Deleting adapters (waits for in-progress operations)
✅ Futures store updates
✅ Base model initialization (one-time, thread-safe)

### What is NOT Thread-Safe (by design)
⚠️ Training the same adapter from multiple threads (serialized automatically)
⚠️ Gradient accumulation across concurrent requests (isolated per request)

## Limitations

### Current Limitations
1. **Megatron backend**: Does not support concurrent processing
2. **GPU memory**: Concurrent adapters share GPU memory - too many concurrent adapters may cause OOM
3. **LoRA state persistence**: Adapters are in-memory only (not persisted across server restarts)

### Recommended Best Practices
1. **Tune MAX_WORKERS**: Start with 4 and increase based on GPU memory
2. **Monitor GPU memory**: Use `nvidia-smi` to monitor VRAM usage
3. **Use small LoRA ranks**: Smaller ranks (r=4-16) allow more concurrent adapters
4. **Avoid concurrent ops on same adapter**: Design clients to use unique adapters per user

## Implementation Details

### Key Changes

#### `hf_backend.py`
- Added `_base_model_lock`, `_adapters_lock`, `_adapter_locks`
- Implemented `_get_adapter_lock()` context manager for per-adapter locking
- Updated all functions to use appropriate locks:
  - `initialize_base_model()`: Double-checked locking pattern
  - `create_lora_adapter()`: Adapter dict lock + existence check
  - `forward_backward()`: Per-adapter lock for training
  - `optim_step()`: Per-adapter lock for optimizer
  - `generate()`: Per-adapter lock for inference
  - `delete_adapter()`: Adapter lock + cleanup

#### `worker.py`
- Created `ThreadPoolExecutor` with configurable `MAX_WORKERS`
- Split job processing into two paths:
  - `_process_job()`: Thread-safe HF backend operations
  - `_process_megatron_job()`: Sequential Megatron operations
- Added `_futures_store_lock` for thread-safe status updates
- Worker loop submits jobs to thread pool instead of processing inline

#### `app.py`
- Added `futures_store_lock` for thread-safe futures access
- Updated all endpoints to use lock when accessing futures_store
- Shared lock with worker module via `worker._futures_store_lock`

## Future Improvements

### Planned Enhancements
1. **Megatron concurrent support**: Multiple model instances for concurrent training
2. **GPU memory management**: Automatic adapter eviction when GPU memory is low
3. **Persistent adapter storage**: Save/load adapters from disk/S3
4. **Request batching**: Batch multiple inference requests for efficiency
5. **Priority queues**: Prioritize certain users or job types

## Troubleshooting

### Issue: Deadlock or Hanging Requests
**Cause**: Potential lock ordering issue or infinite wait
**Solution**: Check logs for exceptions, restart server

### Issue: GPU Out of Memory
**Cause**: Too many concurrent adapters
**Solution**: Reduce `MAX_WORKERS` or use smaller LoRA ranks

### Issue: Slow Performance Despite Concurrency
**Cause**: All requests targeting the same adapter (serialized)
**Solution**: Ensure users are using different adapter IDs

### Issue: "LoRA adapter not found" errors
**Cause**: Adapter deleted while in use, or race condition
**Solution**: Check adapter lifecycle, ensure proper cleanup

## Monitoring

### Recommended Monitoring
1. **GPU utilization**: Should be high (70-100%) with concurrent users
2. **Thread pool activity**: Monitor active threads in logs
3. **Request latency**: Should decrease with concurrent processing
4. **Error rates**: Watch for timeout or OOM errors

### Logging
The server logs key events:
- `Worker pool size: N threads` - Worker pool initialization
- `Created LoRA adapter 'X'` - Adapter creation
- `[model_id] Forward-backward complete` - Training step completion
- `[model_id] Optimizer step complete` - Optimizer step completion

## Conclusion

The Flask Tinker server now supports efficient concurrent processing of multiple users with different LoRA adapters, providing significant performance improvements for multi-tenant scenarios.

For questions or issues, please check the troubleshooting section or review the test script for usage examples.
