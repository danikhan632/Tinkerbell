# Flask Server Quick Start Guide

Get up and running with the Flask server in 5 minutes!

## Prerequisites

```bash
pip install torch transformers peft accelerate flask celery redis requests
```

Optional for Tinker SDK:
```bash
pip install tinker-sdk
```

## Step 1: Start Redis (for Celery)

The server uses Celery for async job processing, which requires Redis:

```bash
# Install Redis if needed
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis

# Start Redis
redis-server
```

Or use Docker:
```bash
docker run -d -p 6379:6379 redis:latest
```

## Step 2: Start the Flask Server

```bash
cd /home/green/code/thinker/flask_server
python app.py
```

You should see:
```
Initializing HuggingFace backend...
Loading base model from HuggingFaceTB/SmolLM2-135M-Instruct...
Base model loaded on cuda
 * Running on http://0.0.0.0:8000
```

## Step 3: Run an Example

### Option A: Simple Example (No Tinker SDK)

In a new terminal:

```bash
cd /home/green/code/thinker/flask_server
python example_simple.py
```

This will:
1. Check server health
2. List available loss functions
3. Train a model with supervised learning (cross_entropy)
4. Train a model with RL (PPO)
5. List all trained adapters

### Option B: Tinker SDK Example

```bash
pip install tinker-sdk
python example_tinker_client.py
```

## Understanding the Output

### Supervised Learning (cross_entropy)

```
--- Step 1 ---
Loss: 628.1056
Metrics:
  - mean_nll: 13.6545
  - perplexity: 937017.9688
  - num_tokens: 46.0
  - num_weighted_tokens: 46.0
```

- **Loss**: Sum of negative log-likelihoods
- **mean_nll**: Average negative log-likelihood per token
- **perplexity**: exp(mean_nll) - lower is better
- **num_tokens**: Total tokens processed
- **num_weighted_tokens**: Tokens used for training (excludes padding)

### RL Training (PPO)

```
--- Step 1 ---
Loss: -15.67
PPO Metrics:
  - Mean advantage: 0.8560
  - Mean ratio: 1.0500
  - Clip fraction: 0.1500
```

- **Mean advantage**: Average reward advantage
- **Mean ratio**: Average importance ratio p_θ(x)/q(x)
- **Clip fraction**: Fraction of ratios that were clipped (prevents large updates)

## Testing the API Manually

### List Loss Functions

```bash
curl http://localhost:8000/loss_functions
```

Response:
```json
{
  "available_loss_functions": ["cross_entropy", "importance_sampling", "ppo"],
  "descriptions": {
    "cross_entropy": "Standard supervised learning (negative log-likelihood)",
    "importance_sampling": "REINFORCE with importance sampling for RL",
    "ppo": "Proximal Policy Optimization with clipping for RL"
  }
}
```

### Train a Model

```bash
# Submit training job
curl -X POST http://localhost:8000/fwdbwd \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "data": [[
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"}
    ]],
    "loss_fn": "cross_entropy"
  }'

# Response: {"request_id": "future_abc123", "model_id": "test-model"}

# Retrieve result
curl -X POST http://localhost:8000/retrieve_future \
  -H "Content-Type: application/json" \
  -d '{"request_id": "future_abc123"}'
```

### List Adapters

```bash
curl http://localhost:8000/adapters
```

Response:
```json
{
  "adapters": [
    {
      "model_id": "test-model",
      "trainable_params": 921600,
      "has_gradients": false
    }
  ]
}
```

## Using Megatron Backend (Advanced)

To use Megatron-LM instead of HuggingFace:

```bash
export USE_MEGATRON=true
python app.py
```

Requirements:
- Megatron-LM installed
- Multi-GPU setup
- Torch distributed initialized

## Troubleshooting

### "Connection refused" error

Make sure Redis is running:
```bash
redis-cli ping
# Should return: PONG
```

### "CUDA out of memory"

Reduce batch size or use CPU:
```python
# In hf_backend.py, line 36:
device = torch.device("cpu")  # Force CPU
```

### Jobs stuck in "pending"

Check the worker thread logs in the server output. The worker may have crashed.

Restart the server:
```bash
# Ctrl+C to stop
python app.py
```

### Import errors

Make sure all dependencies are installed:
```bash
pip install torch transformers peft accelerate flask celery redis requests
```

## Next Steps

1. **Read the full documentation**: `INTEGRATION_README.md`
2. **Explore loss functions**: See `../LOSS_FUNCTIONS.md`
3. **Customize LoRA config**: Edit `hf_backend.py`, `LoraConfigParams` class
4. **Add custom loss functions**: See `loss_functions.py`, `register_custom_loss()`
5. **Deploy models**: Implement `save_weights_for_sampler()` in worker.py

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Flask Server (app.py)                │
│  Routes: /fwdbwd, /optim_step, /loss_functions, etc.   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ Celery Worker   │
            │  (worker.py)    │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  HuggingFace     │    │   Megatron-LM    │
│  Backend         │    │   Backend        │
│ (hf_backend.py)  │    │ (worker.py)      │
│                  │    │                  │
│ - LoRA adapters  │    │ - Distributed    │
│ - Loss functions │    │ - Tensor/Pipeline│
│ - Multi-model    │    │   parallelism    │
└──────────────────┘    └──────────────────┘
```

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/healthz` | GET | Check server health |
| `/loss_functions` | GET | List available loss functions |
| `/fwdbwd` | POST | Forward-backward pass |
| `/optim_step` | POST | Apply optimizer step |
| `/retrieve_future` | POST | Get async job result |
| `/adapters` | GET | List LoRA adapters |
| `/adapters/<id>` | DELETE | Delete adapter |
| `/generate` | POST | Generate text |

## Support

- Documentation: See `INTEGRATION_README.md`
- Examples: `example_simple.py`, `example_tinker_client.py`, `example_tinker_full.py`
- Loss functions: `../LOSS_FUNCTIONS.md`
- Issues: Report bugs in the GitHub repository
