# Tinkerbell Async API Guide

## Overview

Tinkerbell provides **both sync and async** methods for all operations, following the [Tinker async pattern](https://tinker-docs.thinkingmachines.ai/async).

**Key Difference:**
- **Sync endpoints** - Wait for results before returning (blocking)
- **Async endpoints** - Return futures immediately (non-blocking)

## Quick Comparison

| Operation | Sync Endpoint | Async Endpoint |
|-----------|---------------|----------------|
| Forward | `/api/v1/forward` | `/api/v1/forward_async` |
| Forward-Backward | `/api/v1/forward_backward` | `/api/v1/forward_backward_async` |
| Optimizer Step | `/api/v1/optim_step` | `/api/v1/optim_step_async` |
| Add LoRA | `/api/v1/add_lora` | `/api/v1/add_lora_async` |
| Remove LoRA | `/api/v1/remove_lora` | `/api/v1/remove_lora_async` |
| Sample | `/api/v1/sample` | `/api/v1/asample` |

---

## Sync vs. Async Pattern

### Sync Pattern (Simple, Blocking)

**Use when:** You want simple, linear code flow

```python
import requests

# 1. Submit request - BLOCKS until complete
response = requests.post("http://localhost:8000/api/v1/forward_backward", json={
    "model_id": "user_123",
    "data": training_data,
    "loss_fn": "cross_entropy"
})

# 2. Get result immediately
result = response.json()
print(f"Loss: {result['loss']}")
```

**Pros:** Simple, easy to understand
**Cons:** Blocks thread while waiting

---

### Async Pattern (Parallel, Non-Blocking)

**Use when:** You want maximum throughput with concurrent requests

```python
import requests

# 1. Submit request - Returns IMMEDIATELY
response = requests.post("http://localhost:8000/api/v1/forward_backward_async", json={
    "model_id": "user_123",
    "data": training_data,
    "loss_fn": "cross_entropy"
})

future_id = response.json()["request_id"]

# 2. Do other work while training runs...
# Submit more requests, process data, etc.

# 3. Retrieve result when ready
result_response = requests.post("http://localhost:8000/retrieve_future", json={
    "request_id": future_id
})

if result_response.status_code == 202:
    print("Still pending...")
elif result_response.status_code == 200:
    result = result_response.json()
    print(f"Loss: {result['loss']}")
```

**Pros:** Maximum throughput, non-blocking
**Cons:** More complex code

---

## Performance Optimization

**Recommended pattern for high throughput:**

```python
# Submit multiple requests in parallel
future1 = submit_async(batch1)
future2 = submit_async(batch2)
future3 = submit_async(batch3)

# All three are processing concurrently!

# Retrieve results
result1 = get_future(future1)
result2 = get_future(future2)
result3 = get_future(future3)
```

This prevents missing Tinkerbell's ~10 second training cycles by submitting your next request while the current one runs.

---

## Complete Examples

### Example 1: Sync Training (Simple)

```python
#!/usr/bin/env python3
"""Sync training - simple and straightforward."""
import requests

BASE_URL = "http://localhost:8000"

# 1. Create LoRA adapter (sync)
response = requests.post(f"{BASE_URL}/api/v1/add_lora", json={
    "base_model": "base",
    "rank": 16,
    "alpha": 32
})
model_id = response.json()["model_id"]
print(f"Created adapter: {model_id}")

# 2. Training data
training_data = [[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]]

# 3. Train (sync - blocks until complete)
for step in range(5):
    print(f"Step {step+1}/5...")

    # Forward-backward (blocks)
    response = requests.post(f"{BASE_URL}/api/v1/forward_backward", json={
        "model_id": model_id,
        "data": training_data,
        "loss_fn": "cross_entropy"
    })
    loss = response.json()["loss"]
    print(f"  Loss: {loss:.4f}")

    # Optimizer step (blocks)
    requests.post(f"{BASE_URL}/api/v1/optim_step", json={
        "model_id": model_id,
        "adam_params": {"learning_rate": 0.001}
    })
    print(f"  Optimizer applied")

print("Done!")
```

---

### Example 2: Async Training (High Throughput)

```python
#!/usr/bin/env python3
"""Async training - maximum throughput."""
import requests
import time

BASE_URL = "http://localhost:8000"

def retrieve_future(future_id, timeout=60):
    """Wait for future to complete and return result."""
    start = time.time()
    while True:
        response = requests.post(f"{BASE_URL}/retrieve_future", json={
            "request_id": future_id
        })

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            # Still pending
            if time.time() - start > timeout:
                raise TimeoutError(f"Future {future_id} timed out")
            time.sleep(0.5)
        else:
            raise Exception(f"Error: {response.text}")

# 1. Create LoRA adapter
response = requests.post(f"{BASE_URL}/api/v1/add_lora_async", json={
    "base_model": "base",
    "rank": 16,
    "alpha": 32
})
future_id = response.json()["request_id"]
model_id = retrieve_future(future_id)["model_id"]
print(f"Created adapter: {model_id}")

# 2. Training data
training_data = [[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]]

# 3. ASYNC Training - submit all forward-backward passes in parallel!
print("Submitting 5 training steps asynchronously...")
fwdbwd_futures = []

for step in range(5):
    # Submit forward-backward (returns immediately!)
    response = requests.post(f"{BASE_URL}/api/v1/forward_backward_async", json={
        "model_id": model_id,
        "data": training_data,
        "loss_fn": "cross_entropy"
    })
    fwdbwd_futures.append(response.json()["request_id"])
    print(f"  Submitted step {step+1}")

print("All steps submitted! Waiting for results...")

# 4. Retrieve results as they complete
for i, future_id in enumerate(fwdbwd_futures, 1):
    result = retrieve_future(future_id)
    print(f"  Step {i} complete - Loss: {result['loss']:.4f}")

    # Apply optimizer step
    response = requests.post(f"{BASE_URL}/api/v1/optim_step_async", json={
        "model_id": model_id,
        "adam_params": {"learning_rate": 0.001}
    })
    optim_future = response.json()["request_id"]
    retrieve_future(optim_future)
    print(f"  Step {i} optimizer applied")

print("Done!")
```

---

### Example 3: Concurrent Multi-User Training

```python
#!/usr/bin/env python3
"""Train multiple users concurrently using async API."""
import requests
import time
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000"

def retrieve_future(future_id, timeout=60):
    """Wait for future to complete and return result."""
    start = time.time()
    while True:
        response = requests.post(f"{BASE_URL}/retrieve_future", json={
            "request_id": future_id
        })

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            if time.time() - start > timeout:
                raise TimeoutError(f"Future {future_id} timed out")
            time.sleep(0.5)
        else:
            raise Exception(f"Error: {response.text}")

def train_user(user_name):
    """Train a single user's adapter asynchronously."""
    print(f"[{user_name}] Starting...")

    # 1. Create adapter (async)
    response = requests.post(f"{BASE_URL}/api/v1/add_lora_async", json={
        "base_model": "base",
        "rank": 16,
        "alpha": 32
    })
    future_id = response.json()["request_id"]
    result = retrieve_future(future_id)
    model_id = result["model_id"]
    print(f"[{user_name}] Created adapter: {model_id}")

    # 2. Training data
    training_data = [[
        {"role": "user", "content": f"Hello {user_name}!"},
        {"role": "assistant", "content": f"Hi {user_name}!"}
    ]]

    # 3. Train (async)
    for step in range(3):
        # Forward-backward (async)
        response = requests.post(f"{BASE_URL}/api/v1/forward_backward_async", json={
            "model_id": model_id,
            "data": training_data,
            "loss_fn": "cross_entropy"
        })
        future_id = response.json()["request_id"]
        result = retrieve_future(future_id)
        print(f"[{user_name}] Step {step+1}: Loss={result['loss']:.4f}")

        # Optimizer step (async)
        response = requests.post(f"{BASE_URL}/api/v1/optim_step_async", json={
            "model_id": model_id,
            "adam_params": {"learning_rate": 0.001}
        })
        retrieve_future(response.json()["request_id"])

    print(f"[{user_name}] Done!")

# Train 3 users concurrently
with ThreadPoolExecutor(max_workers=3) as executor:
    users = ["Alice", "Bob", "Carol"]
    executor.map(train_user, users)

print("All users trained!")
```

---

## API Reference

### Retrieve Future

**Endpoint:** `POST /retrieve_future`

**Request:**
```json
{
  "request_id": "future_abc123"
}
```

**Response (Pending):** `202 Accepted`
```json
{
  "status": "pending"
}
```

**Response (Complete):** `200 OK`
```json
{
  "loss": 0.5432,
  "num_samples": 1,
  "model_id": "user_123",
  "backend": "megatron-bridge"
}
```

**Response (Error):** `500 Internal Server Error`
```json
{
  "detail": "Job failed: <error message>"
}
```

---

## Endpoint Matrix

| Category | Operation | Sync Endpoint | Async Endpoint |
|----------|-----------|---------------|----------------|
| **Training** | Forward | `/api/v1/forward` | `/api/v1/forward_async` |
|  | Forward-Backward | `/api/v1/forward_backward` | `/api/v1/forward_backward_async` |
|  | Optimizer Step | `/api/v1/optim_step` | `/api/v1/optim_step_async` |
| **LoRA** | Add Adapter | `/api/v1/add_lora` | `/api/v1/add_lora_async` |
|  | Remove Adapter | `/api/v1/remove_lora` | `/api/v1/remove_lora_async` |
| **Sampling** | Sample | `/api/v1/sample` | `/api/v1/asample` |
| **Weights** | Load | N/A | `/api/v1/load_weights_async` |
|  | Save | N/A | `/api/v1/save_weights_async` |
|  | Save for Sampler | N/A | `/api/v1/save_weights_for_sampler_async` |
| **Futures** | Retrieve | `/retrieve_future` | Same |

---

## Legacy Endpoints

For backward compatibility, these endpoints still work:

| Legacy | New Async Equivalent |
|--------|---------------------|
| `/fwd` | `/api/v1/forward_async` |
| `/fwdbwd` | `/api/v1/forward_backward_async` |
| `/optim_step` | `/api/v1/optim_step_async` |
| `/add_lora` | `/api/v1/add_lora_async` |
| `/remove_lora` | `/api/v1/remove_lora_async` |
| `/load_weights` | `/api/v1/load_weights_async` |
| `/save_weights` | `/api/v1/save_weights_async` |

---

## Performance Tips

### Tip 1: Use Async for Training Loops

```python
# ❌ SLOW (sync - sequential)
for batch in batches:
    result = train_sync(batch)  # Waits for each

# ✅ FAST (async - parallel)
futures = [train_async(batch) for batch in batches]
results = [get_future(f) for f in futures]
```

### Tip 2: Submit Next Request While Current Runs

```python
# ❌ SLOW (wastes time between requests)
result1 = train_sync(batch1)  # Training runs
# Gap - nothing happening
result2 = train_sync(batch2)  # Training runs
# Gap - nothing happening

# ✅ FAST (no gaps)
future1 = train_async(batch1)  # Training runs
future2 = train_async(batch2)  # Training runs (while #1 still running!)
result1 = get_future(future1)
result2 = get_future(future2)
```

### Tip 3: Poll Efficiently

```python
# ❌ BAD (too frequent polling)
while True:
    result = check_future(future_id)
    if result: break
    time.sleep(0.01)  # 100x per second!

# ✅ GOOD (reasonable polling)
while True:
    result = check_future(future_id)
    if result: break
    time.sleep(0.5)  # 2x per second
```

---

## Testing

### Check Server Endpoints
```bash
curl http://localhost:8000/ | python3 -m json.tool
```

Should show all sync and async endpoints.

### Test Async Endpoint
```bash
# 1. Submit async request
curl -X POST http://localhost:8000/api/v1/forward_backward_async \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test",
    "data": [[{"role": "user", "content": "test"}]],
    "loss_fn": "cross_entropy"
  }'

# Returns: {"request_id": "future_abc123", "model_id": "test"}

# 2. Retrieve result
curl -X POST http://localhost:8000/retrieve_future \
  -H "Content-Type: application/json" \
  -d '{"request_id": "future_abc123"}'

# Returns: {"status": "pending"} or {"loss": 0.5, ...}
```

---

## Summary

**Choose Sync when:**
- Simple scripts
- Learning/testing
- Linear workflow
- Don't care about max performance

**Choose Async when:**
- Production systems
- Maximum throughput
- Concurrent operations
- Processing batches
- Multi-user training

**Key Pattern:**
```python
# Submit requests as fast as possible
futures = [submit_async(batch) for batch in batches]

# Retrieve results
results = [get_future(f) for f in futures]
```

This maximizes GPU utilization and minimizes wait time!
