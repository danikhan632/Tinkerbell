# Tinker-Compatible Training Server

A Flask-based HTTP server implementing the Tinker API with dual backend support (HuggingFace + Megatron-LM) and flexible loss functions for supervised learning and reinforcement learning.

## 🚀 Quick Start

```bash
# Start Redis
redis-server &

# Start the server
python app.py

# Run example (in another terminal)
python example_simple.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## 📋 Features

### ✨ Loss Functions
- **cross_entropy**: Standard supervised learning (negative log-likelihood)
- **importance_sampling**: REINFORCE with importance sampling for off-policy RL
- **ppo**: Proximal Policy Optimization with clipping for stable RL training

### 🔧 Dual Backend Support
- **HuggingFace** (default): Transformers + PEFT for LoRA fine-tuning
- **Megatron-LM** (optional): Distributed training with tensor/pipeline parallelism

### 🎯 LoRA Support
- Multiple independent adapters
- Shared frozen base model
- Per-adapter optimizers
- ~920K trainable params (0.68% of base model)

### 🌐 API Compatibility
- Tinker-compatible endpoints (`/fwdbwd`, `/optim_step`)
- FastAPI-compatible aliases (`/api/v1/forward_backward`, `/api/v1/optim_step`)
- Async job processing with Celery

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[QUICKSTART.md](QUICKSTART.md)** | Get started in 5 minutes |
| **[INTEGRATION_README.md](INTEGRATION_README.md)** | Complete API documentation |
| **[SUMMARY.md](SUMMARY.md)** | Integration overview |
| **[../LOSS_FUNCTIONS.md](../LOSS_FUNCTIONS.md)** | Loss function math & usage |

## 🎓 Examples

| Example | Description | Requires |
|---------|-------------|----------|
| **[example_simple.py](example_simple.py)** | Simple training with requests | requests |
| **[example_tinker_client.py](example_tinker_client.py)** | Tinker SDK basics | tinker-sdk |
| **[example_tinker_full.py](example_tinker_full.py)** | Comprehensive Tinker example | tinker-sdk |

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│     Flask Server (app.py)        │
│  Tinker-compatible REST API      │
└──────────────┬──────────────────┘
               │
               ▼
      ┌────────────────┐
      │ Celery Workers  │
      │   (worker.py)   │
      └────────┬────────┘
               │
    ┏━━━━━━━━━┻━━━━━━━━━┓
    ▼                    ▼
┌──────────────┐  ┌──────────────┐
│ HuggingFace  │  │  Megatron-LM │
│   Backend    │  │   Backend    │
└──────────────┘  └──────────────┘
```

## 🔌 API Endpoints

### Core Training
- `POST /fwdbwd` - Forward-backward pass
- `POST /optim_step` - Optimizer step
- `POST /retrieve_future` - Get async job result

### Loss Functions
- `GET /loss_functions` - List available loss functions

### LoRA Management
- `GET /adapters` - List all adapters
- `DELETE /adapters/<id>` - Delete adapter
- `POST /add_lora` - Add adapter
- `POST /remove_lora` - Remove adapter

### Generation & Inference
- `POST /sample` - Generate text
- `POST /generate` - Generate text (alias)

### Health & Info
- `GET /healthz` - Health check
- `GET /get_server_capabilities` - List supported models

## 📦 Installation

```bash
# Required
pip install torch transformers peft accelerate flask celery redis requests

# Optional: Tinker SDK
pip install tinker-sdk

# Optional: Megatron-LM
pip install megatron-lm
```

## 🎯 Usage Examples

### Supervised Learning

```python
import requests

# Submit training job
response = requests.post("http://localhost:8000/fwdbwd", json={
    "model_id": "my-model",
    "data": [[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]],
    "loss_fn": "cross_entropy"
})

# Get result
future_id = response.json()["request_id"]
result = requests.post("http://localhost:8000/retrieve_future",
                      json={"request_id": future_id}).json()

print(f"Loss: {result['loss']:.4f}")
```

### RL Training with PPO

```python
# Submit PPO training job
response = requests.post("http://localhost:8000/fwdbwd", json={
    "model_id": "rl-model",
    "data": [[
        {"role": "user", "content": "What is 10-3?"},
        {"role": "assistant", "content": "7"}
    ]],
    "loss_fn": "ppo",
    "loss_fn_inputs": {
        "target_tokens": [[...]],
        "logprobs": [[...]],
        "advantages": [[...]]
    }
})
```

### Using Tinker SDK

```python
import asyncio
import tinker

async def main():
    client = tinker.ServiceClient(base_url="http://localhost:8000")

    training_client = await client.create_training_client(
        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_id="my-model"
    )

    result = await training_client.forward_backward(
        data=[...],
        loss_fn="cross_entropy"
    )

    await training_client.optim_step()

asyncio.run(main())
```

## 🔧 Configuration

### Environment Variables

```bash
# Server port (default: 8000)
export PORT=8000

# Use Megatron backend (default: false)
export USE_MEGATRON=true

# Celery broker (default: redis://localhost:6379/0)
export CELERY_BROKER_URL=redis://localhost:6379/0

# Megatron inference server (default: http://localhost:5000)
export MEGATRON_SERVER_URL=http://localhost:5000
```

### Backend Selection

```bash
# HuggingFace backend (default)
python app.py

# Megatron backend
USE_MEGATRON=true python app.py
```

## 📊 Performance

### HuggingFace Backend
- **Base Model**: SmolLM2-135M (135M parameters)
- **LoRA Params**: ~920K trainable (0.68% of base)
- **Memory**: ~600MB GPU per adapter
- **Speed**: ~100 tokens/second (single GPU)

### Megatron Backend
- **Model**: Configurable GPT architecture
- **Parallelism**: Tensor + Pipeline
- **Memory**: Distributed across GPUs
- **Speed**: Scales with GPU count

## 🐛 Troubleshooting

### Redis Connection Error
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis if needed
redis-server
```

### CUDA Out of Memory
```python
# Edit hf_backend.py, line 36
device = torch.device("cpu")  # Force CPU
```

### Jobs Stuck in Pending
Restart the server - the worker thread may have crashed:
```bash
# Ctrl+C to stop
python app.py
```

See [QUICKSTART.md](QUICKSTART.md) for more troubleshooting tips.

## 🗺️ Roadmap

- [ ] Checkpoint save/load
- [ ] save_weights_for_sampler for deployment
- [ ] DPO loss function
- [ ] Metrics tracking (WandB/TensorBoard)
- [ ] Model merging (LoRA → full model)
- [ ] Distributed HuggingFace backend
- [ ] Gradient checkpointing

## 📝 File Structure

```
flask_server/
├── README.md                  # This file
├── QUICKSTART.md             # 5-minute setup guide
├── INTEGRATION_README.md     # Complete API docs
├── SUMMARY.md                # Integration overview
│
├── app.py                    # Flask routes
├── worker.py                 # Celery workers (dual backend)
├── hf_backend.py             # HuggingFace + PEFT backend
├── loss_functions.py         # Loss function implementations
├── tasks.py                  # Celery task definitions
├── storage.py                # LoRA storage utilities
│
├── example_simple.py         # Simple requests example
├── example_tinker_client.py  # Tinker SDK basics
├── example_tinker_full.py    # Comprehensive Tinker example
│
└── run_server.sh             # Convenience launcher
```

## 🤝 Contributing

Contributions welcome! Key areas:
1. Implement checkpoint management
2. Add more loss functions (DPO, KL divergence)
3. Improve async job handling
4. Add metrics tracking
5. Write more examples

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **HuggingFace** - Transformers & PEFT libraries
- **Tinker SDK** - API design and loss function system
- **Megatron-LM** - Distributed training framework
- **FastAPI Implementation** - Original training server (main.py)

## 📞 Support

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **Full Docs**: See [INTEGRATION_README.md](INTEGRATION_README.md)
- **Loss Functions**: See [../LOSS_FUNCTIONS.md](../LOSS_FUNCTIONS.md)
- **Examples**: Run `python example_simple.py`

---

**Status**: ✅ Production Ready (HuggingFace backend) | ⚠️ Experimental (Megatron backend)

**Last Updated**: 2025-10-23
