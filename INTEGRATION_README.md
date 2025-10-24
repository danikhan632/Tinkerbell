# Flask Server Integration with Loss Functions

This Flask server now supports **dual backends**:
- **Megatron-LM**: Distributed training with tensor/pipeline parallelism
- **HuggingFace + PEFT**: LoRA fine-tuning with flexible loss functions

## Quick Start

### HuggingFace Backend (Default)

```bash
cd /home/green/code/thinker/flask_server
python app.py
```

The server will start on `http://localhost:8000` using the HuggingFace backend by default.

### Megatron Backend (Optional)

To use the Megatron backend, set the environment variable:

```bash
export USE_MEGATRON=true
python app.py
```

## Features

### Loss Functions (HuggingFace Backend)

The HuggingFace backend supports three loss functions:

1. **cross_entropy**: Standard supervised learning
2. **importance_sampling**: REINFORCE with importance sampling for RL
3. **ppo**: Proximal Policy Optimization with clipping for RL

### API Endpoints

#### Core Training Endpoints

- `POST /fwdbwd` or `POST /api/v1/forward_backward`: Forward-backward pass
- `POST /optim_step` or `POST /api/v1/optim_step`: Optimizer step
- `POST /retrieve_future`: Retrieve async job results

#### Loss Functions

- `GET /loss_functions`: List available loss functions

Example response:
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

#### LoRA Management

- `GET /adapters`: List all LoRA adapters
- `DELETE /adapters/<model_id>`: Delete a specific adapter
- `POST /add_lora`: Add a new LoRA adapter
- `POST /remove_lora`: Remove a LoRA adapter

#### Generation

- `POST /api/v1/sample` or `POST /generate`: Generate text with a specific adapter

#### Health & Info

- `GET /healthz`: Health check
- `GET /get_server_capabilities`: List supported models
- `POST /get_info`: Get model/adapter info

## Training with Loss Functions

### Example 1: Supervised Learning

```python
import requests

BASE_URL = "http://localhost:8000"

# Step 1: Forward-backward with cross_entropy loss
fwdbwd_resp = requests.post(f"{BASE_URL}/api/v1/forward_backward", json={
    "model_id": "my-model",
    "data": [[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]],
    "loss_fn": "cross_entropy"
})

future_id = fwdbwd_resp.json()["request_id"]

# Step 2: Retrieve result
retrieve_resp = requests.post(f"{BASE_URL}/retrieve_future", json={
    "request_id": future_id
})

result = retrieve_resp.json()
print(f"Loss: {result['loss']:.4f}")
print(f"Metrics: {result['metrics']}")

# Step 3: Optimizer step
optim_resp = requests.post(f"{BASE_URL}/api/v1/optim_step", json={
    "model_id": "my-model",
    "adam_params": {"learning_rate": 5e-4}
})

optim_future = optim_resp.json()["request_id"]
```

### Example 2: RL Training with PPO

```python
# Step 1: Forward-backward with PPO loss
fwdbwd_resp = requests.post(f"{BASE_URL}/api/v1/forward_backward", json={
    "model_id": "rl-model",
    "data": [[
        {"role": "user", "content": "What is 10-3?"},
        {"role": "assistant", "content": "10-3=7"}
    ]],
    "loss_fn": "ppo",
    "loss_fn_inputs": {
        "target_tokens": [[220, 16, 15, 489, 220, 18, 284, 220, 22]],
        "logprobs": [[-0.4, -0.5, -0.3, -0.4, -0.3, -0.5, -0.2, -0.4, -0.3]],
        "advantages": [[0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0]]
    }
})

# Step 2-3: Retrieve result and apply optimizer step
```

## Architecture

### File Structure

```
flask_server/
├── app.py                    # Flask routes and server setup
├── worker.py                 # Dual backend worker (Megatron + HuggingFace)
├── hf_backend.py            # HuggingFace + PEFT training logic
├── loss_functions.py        # Loss function implementations
├── tasks.py                 # Celery task definitions
├── storage.py               # LoRA weights storage
└── INTEGRATION_README.md    # This file
```

### Backend Selection

The backend is chosen at startup based on the `USE_MEGATRON` environment variable:

- **`USE_MEGATRON=false`** (default): HuggingFace backend with PEFT LoRA
- **`USE_MEGATRON=true`**: Megatron-LM backend for distributed training

### HuggingFace Backend Details

- **Base Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (configurable)
- **LoRA Config**: r=16, lora_alpha=32 (configurable)
- **Device**: Automatically detects CUDA/CPU
- **Multi-LoRA**: Supports multiple independent adapters with shared base model

### Loss Function System

Loss functions are implemented in `loss_functions.py` with a registry pattern:

```python
from flask_server import loss_functions

# Get a loss function
loss_fn = loss_functions.LOSS_REGISTRY.get("ppo")

# List available functions
available = loss_functions.LOSS_REGISTRY.list_available()

# Register custom loss
def my_custom_loss(model_outputs, loss_fn_inputs, attention_mask):
    # Your implementation
    ...

loss_functions.register_custom_loss("my_loss", my_custom_loss)
```

## Migration from FastAPI (main.py)

If you were using the original `main.py` FastAPI server, the Flask server provides the same functionality with additional async support via Celery:

### Key Differences

1. **Async Operations**: All training operations return a `future_id` that you must retrieve with `/retrieve_future`
2. **Endpoint Names**: Both Tinker-compatible (`/fwdbwd`) and FastAPI-compatible (`/api/v1/forward_backward`) routes
3. **Backend Choice**: Can use either Megatron or HuggingFace via environment variable

### Example Conversion

**FastAPI (main.py) - Synchronous:**
```python
response = requests.post("http://localhost:8000/api/v1/forward_backward", json={
    "data": [...],
    "model_id": "my-model"
})
result = response.json()  # Direct result
```

**Flask (app.py) - Asynchronous:**
```python
# Step 1: Submit job
response = requests.post("http://localhost:8000/api/v1/forward_backward", json={
    "data": [...],
    "model_id": "my-model"
})
future_id = response.json()["request_id"]

# Step 2: Retrieve result
import time
time.sleep(0.5)  # Wait for job to complete
result_response = requests.post("http://localhost:8000/retrieve_future", json={
    "request_id": future_id
})
result = result_response.json()
```

## Environment Variables

- `PORT`: Server port (default: 8000)
- `USE_MEGATRON`: Use Megatron backend instead of HuggingFace (default: false)
- `CELERY_BROKER_URL`: Celery broker URL (default: redis://localhost:6379/0)
- `MEGATRON_SERVER_URL`: Megatron inference server URL (default: http://localhost:5000)

## Dependencies

### HuggingFace Backend
- torch
- transformers
- peft
- accelerate

### Megatron Backend (Optional)
- megatron-lm
- torch.distributed

### Server
- flask
- celery
- redis (for Celery broker)

## Running Examples

We provide three example scripts:

### 1. Simple Example (Recommended for Getting Started)

Uses plain `requests` library - no Tinker SDK required:

```bash
# Start the server
cd /home/green/code/thinker/flask_server
python app.py

# In another terminal
python example_simple.py
```

This example shows:
- Health checks
- Listing loss functions
- Training with cross_entropy (supervised learning)
- Training with PPO (reinforcement learning)
- Listing trained adapters

### 2. Tinker Client Example

Uses the official Tinker SDK with async/await:

```bash
# Install Tinker SDK
pip install tinker-sdk

# Run example
python example_tinker_client.py
```

### 3. Full Tinker Example

Comprehensive example showing all Tinker SDK features:

```bash
python example_tinker_full.py
```

This demonstrates:
- Creating training clients
- Training with different loss functions
- Saving models for deployment
- Inference with trained models

## Example Output

```
=== Flask Server Example ===

Server status: ok

Available loss functions:
  - cross_entropy: Standard supervised learning (negative log-likelihood)
  - importance_sampling: REINFORCE with importance sampling for RL
  - ppo: Proximal Policy Optimization with clipping for RL

=== Example 1: Supervised Learning (cross_entropy) ===

--- Step 1 ---
Submitted forward-backward job: future_a1b2c3d4
Loss: 628.1056
Metrics:
  - mean_nll: 13.6545
  - perplexity: 937017.9688
Optimizer step complete

--- Step 2 ---
Loss: 580.3505
Metrics:
  - mean_nll: 12.6163
  - perplexity: 325338.6094
...
```

See `LOSS_FUNCTIONS.md` in the root directory for detailed loss function documentation.
