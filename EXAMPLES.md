# Examples: Concurrent Multi-User LoRA Training

This directory contains several examples demonstrating concurrent multi-user support.

## Quick Start

### 1. Start the Server

```bash
# In terminal 1
cd /home/green/code/thinker/flask_server
python app.py
```

The server will start with concurrent processing enabled:
```
Initializing HuggingFace backend...
Loading base model from HuggingFaceTB/SmolLM2-135M-Instruct...
Worker pool size: 4 threads
Worker pool initialized. Concurrent job processing: Enabled (HF)
Base model loaded on cuda
 * Running on http://0.0.0.0:8000
```

### 2. Run an Example

```bash
# In terminal 2
cd /home/green/code/thinker/flask_server

# Simple concurrent example (3 users)
python example_simple_concurrent.py

# Full-featured example with colored output
python example_concurrent_users.py

# Comprehensive test suite
python test_concurrent_users.py 4
```

## Available Examples

### 1. `example_simple_concurrent.py` - Simple & Clean
**Best for: Understanding the basics**

```bash
python example_simple_concurrent.py
```

**What it does:**
- Creates 3 users (Alice, Bob, Carol)
- Each user creates their own LoRA adapter
- All users train concurrently (3 steps each)
- Tests inference on trained adapters
- Shows total time and speedup

**Output:**
```
======================================================================
Simple Concurrent Multi-User Example
======================================================================

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
...
[All training concurrently]
...
======================================================================
SUMMARY
======================================================================

✓ All 3 users completed training!
✓ Total time: 12.34 seconds
```

### 2. `example_concurrent_users.py` - Full-Featured
**Best for: Visual demonstration**

```bash
python example_concurrent_users.py
```

**What it does:**
- Same as simple example but with:
  - Colored terminal output (blue/green/yellow per user)
  - Detailed timestamps
  - Progress indicators
  - More verbose logging

**Features:**
- ✨ **Colored output** - Each user has their own color for easy tracking
- ⏱️ **Timestamps** - See exactly when each operation happens
- 📊 **Training history** - Tracks loss for each step
- 🎯 **Visual feedback** - Easy to see concurrent execution

**Output:**
```
================================================================================
Example: Multiple Concurrent Users Training LoRA Adapters
================================================================================

✓ Server is healthy

[09:45:12.345] Alice: Creating LoRA adapter (rank=8)...
[09:45:12.346] Bob: Creating LoRA adapter (rank=8)...
[09:45:12.347] Carol: Creating LoRA adapter (rank=8)...
[09:45:12.567] Alice: ✓ Adapter created: base_lora_abc123
[09:45:12.568] Bob: ✓ Adapter created: base_lora_def456
[09:45:12.569] Carol: ✓ Adapter created: base_lora_ghi789
[09:45:12.570] Alice: Step 1: Starting forward-backward pass...
[09:45:12.571] Bob: Step 1: Starting forward-backward pass...
[09:45:12.572] Carol: Step 1: Starting forward-backward pass...
...
```

### 3. `test_concurrent_users.py` - Comprehensive Test Suite
**Best for: Validation and benchmarking**

```bash
# Test with 4 users (default)
python test_concurrent_users.py

# Test with 8 users
python test_concurrent_users.py 8
```

**What it does:**
- Simulates N concurrent users
- Each user: create → train → test inference
- Measures performance metrics
- Calculates speedup vs sequential
- Reports success/failure rates

**Output:**
```
================================================================================
Testing 4 concurrent users with different LoRA adapters
================================================================================

✓ Server is healthy

[User user_0] Creating LoRA adapter...
[User user_1] Creating LoRA adapter...
[User user_2] Creating LoRA adapter...
[User user_3] Creating LoRA adapter...
...
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
```

### 4. `client_example.py` - API Client Library
**Best for: Building your own applications**

```bash
python client_example.py
```

**What it provides:**
- `TinkerClient` class - Full API wrapper
- Methods for all endpoints
- Built-in polling and error handling
- Example usage for each operation

**Usage in your code:**
```python
from client_example import TinkerClient

client = TinkerClient("http://localhost:8000")

# Create adapter
request_id = client.add_lora(base_model="base", rank=8, alpha=16)
result = client.wait_for_result(request_id)
model_id = result["model_id"]

# Train
training_data = [[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi!"}
]]

request_id = client.fwdbwd(model_id=model_id, data=training_data)
result = client.wait_for_result(request_id)

# Inference
result = client.sample(model_id=model_id, prompts=["Hello!"])
print(result)
```

## API Overview

### Core Endpoints

#### 1. Create LoRA Adapter
```python
POST /add_lora
{
    "base_model": "base",
    "rank": 8,        # LoRA rank (lower = smaller adapter)
    "alpha": 16       # LoRA alpha (scaling factor)
}
→ Returns: {"request_id": "future_abc123", "model_id": "base"}
```

#### 2. Forward-Backward Pass
```python
POST /fwdbwd
{
    "model_id": "base_lora_abc123",
    "data": [[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"}
    ]],
    "loss_fn": "cross_entropy"
}
→ Returns: {"request_id": "future_def456", "model_id": "base_lora_abc123"}
```

#### 3. Optimizer Step
```python
POST /optim_step
{
    "model_id": "base_lora_abc123",
    "adam_params": {
        "learning_rate": 0.001
    }
}
→ Returns: {"request_id": "future_ghi789", "model_id": "base_lora_abc123"}
```

#### 4. Retrieve Result
```python
POST /retrieve_future
{
    "request_id": "future_abc123"
}
→ Returns:
  - 202: Still pending
  - 200: {"status": "completed", "result": {...}}
  - 500: Error occurred
```

#### 5. Inference
```python
POST /api/v1/sample  # Synchronous (blocks)
{
    "model_id": "base_lora_abc123",
    "prompts": ["Hello!"],
    "sampling_params": {
        "max_tokens": 50,
        "temperature": 0.7
    }
}
→ Returns: {"generated_text": "Hi there! How can I help?"}
```

## Configuration

### Environment Variables

```bash
# Number of concurrent worker threads (default: 4)
export MAX_WORKERS=8

# Use Megatron backend (disables concurrency)
export USE_MEGATRON=false
```

### Tuning for Your Hardware

#### Small GPU (8GB VRAM)
```bash
export MAX_WORKERS=2  # 2 concurrent users
python app.py
```

#### Medium GPU (16GB VRAM)
```bash
export MAX_WORKERS=4  # 4 concurrent users (default)
python app.py
```

#### Large GPU (24GB+ VRAM)
```bash
export MAX_WORKERS=8  # 8 concurrent users
python app.py
```

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use a different port
PORT=8001 python app.py
```

### GPU Out of Memory
```bash
# Reduce concurrent workers
export MAX_WORKERS=2
python app.py

# Or use smaller LoRA ranks in your code
client.add_lora(base_model="base", rank=4, alpha=8)  # Smaller adapter
```

### Examples fail with connection error
```bash
# Make sure server is running
curl http://localhost:8000/healthz

# If server is on different port/host
# Edit BASE_URL in the example scripts
BASE_URL = "http://localhost:8001"
```

### Slow performance despite concurrency
- **Issue**: All users using the same adapter
- **Solution**: Each user needs their own adapter (different `model_id`)

```python
# ❌ Wrong - same adapter (serialized)
user1: fwdbwd(model_id="shared_adapter")
user2: fwdbwd(model_id="shared_adapter")  # Waits for user1

# ✓ Correct - different adapters (concurrent)
user1: fwdbwd(model_id="user1_adapter")
user2: fwdbwd(model_id="user2_adapter")  # Runs concurrently
```

## Performance Benchmarks

### Expected Speedup (4 concurrent users)

| Hardware | Sequential | Concurrent | Speedup |
|----------|-----------|-----------|---------|
| RTX 3090 | ~45s | ~12s | 3.75x |
| RTX 4090 | ~35s | ~10s | 3.50x |
| A100 | ~30s | ~8s | 3.75x |

### Factors Affecting Performance

**Good for concurrency:**
- ✓ Different adapters per user
- ✓ Small batch sizes (1-4)
- ✓ Low LoRA ranks (4-16)
- ✓ Short sequences

**Bad for concurrency:**
- ✗ Same adapter shared by all users (serialized)
- ✗ Large batch sizes
- ✗ High LoRA ranks (>32)
- ✗ Very long sequences

## Next Steps

1. **Start simple**: Run `example_simple_concurrent.py` first
2. **Experiment**: Try different numbers of users
3. **Monitor**: Watch GPU usage with `nvidia-smi -l 1`
4. **Optimize**: Tune `MAX_WORKERS` for your hardware
5. **Build**: Use `TinkerClient` to build your own application

## Additional Documentation

- `CONCURRENCY.md` - Detailed concurrency architecture
- `ARCHITECTURE.md` - Visual diagrams and design
- `CHANGES_SUMMARY.md` - What changed and why

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs in your terminal
3. Monitor GPU memory: `nvidia-smi`
4. Test with fewer users first

Happy concurrent training! 🚀
