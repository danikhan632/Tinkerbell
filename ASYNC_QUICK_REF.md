# Async API Quick Reference

## 🎯 Quick Decision

**Want simple code?** → Use **Sync** endpoints
**Want max speed?** → Use **Async** endpoints

---

## 📖 Endpoint Cheat Sheet

```
# SYNC (waits for result)
POST /api/v1/forward_backward
POST /api/v1/optim_step
POST /api/v1/add_lora
POST /api/v1/remove_lora
POST /api/v1/sample

# ASYNC (returns future)
POST /api/v1/forward_backward_async
POST /api/v1/optim_step_async
POST /api/v1/add_lora_async
POST /api/v1/remove_lora_async
POST /api/v1/asample

# RETRIEVE RESULTS
POST /retrieve_future
```

---

## 💻 Code Templates

### Sync (Simple)
```python
import requests

r = requests.post("http://localhost:8000/api/v1/forward_backward", json={
    "model_id": "user_123",
    "data": [[{"role": "user", "content": "test"}]],
    "loss_fn": "cross_entropy"
})
loss = r.json()["loss"]  # Got result!
```

### Async (Fast)
```python
import requests, time

def get_future(fid):
    while True:
        r = requests.post("http://localhost:8000/retrieve_future",
                         json={"request_id": fid})
        if r.status_code == 200: return r.json()
        time.sleep(0.5)

# Submit (returns immediately)
r = requests.post("http://localhost:8000/api/v1/forward_backward_async", json={
    "model_id": "user_123",
    "data": [[{"role": "user", "content": "test"}]],
    "loss_fn": "cross_entropy"
})
future_id = r.json()["request_id"]

# Get result later
result = get_future(future_id)
loss = result["loss"]
```

### Batched Async (Fastest!)
```python
# Submit all
futures = []
for batch in batches:
    r = requests.post("http://localhost:8000/api/v1/forward_backward_async",
                     json={"model_id": "user_123", "data": batch, "loss_fn": "cross_entropy"})
    futures.append(r.json()["request_id"])

# Get all results
results = [get_future(f) for f in futures]
```

---

## ⚡ Performance

| Pattern | Time for 5 batches | Speedup |
|---------|-------------------|---------|
| **Sync** | ~15s | 1x |
| **Async** | ~4s | **3-4x** |

---

## 🧪 Quick Test

```bash
# Check endpoints
curl http://localhost:8000/

# Test async
curl -X POST http://localhost:8000/api/v1/forward_backward_async \
  -H "Content-Type: application/json" \
  -d '{"model_id":"test","data":[[{"role":"user","content":"hi"}]],"loss_fn":"cross_entropy"}'

# Get result
curl -X POST http://localhost:8000/retrieve_future \
  -H "Content-Type: application/json" \
  -d '{"request_id":"future_abc123"}'
```

---

## 📚 Full Docs

- **Complete Guide**: `ASYNC_API.md`
- **Examples**: `examples/example_async_training.py`
- **Summary**: `ASYNC_IMPLEMENTATION_SUMMARY.md`
- **Tinker Docs**: https://tinker-docs.thinkingmachines.ai/async
