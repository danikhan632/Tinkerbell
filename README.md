# Tinkerbell 🔔

A toy implementation of the Tinker API for fine-tuning large language models with LoRA adapters and custom loss functions.

## Features

- **Multi-Backend Support**: HuggingFace, Megatron-LM, and vLLM backends
- **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning with PEFT
- **Custom Loss Functions**: Define custom loss functions with a simple decorator (DPO, PPO, contrastive learning, etc.)
- **Built-in Losses**: Cross-entropy, importance sampling, and PPO
- **Concurrent Training**: Multi-threaded worker for concurrent LoRA adapter training
- **Co-located vLLM**: High-performance sampling with LoRA support
- **REST API**: Flask-based server compatible with Tinker SDK

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install vLLM for high-performance sampling
pip install vllm
```

### Start the Server

```bash
python src/app.py
```

The server will start on `http://localhost:8000`.

### Basic Training Example

```python
import sys
sys.path.insert(0, 'src')
import hf_backend

# Initialize backend
hf_backend.initialize_base_model("HuggingFaceTB/SmolLM2-135M-Instruct")

# Create LoRA adapter
lora_config = hf_backend.LoraConfigParams(r=16, lora_alpha=32)
hf_backend.create_lora_adapter("my-model", lora_config)

# Training data
data = [
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
]

# Forward-backward pass
result = hf_backend.forward_backward(
    model_id="my-model",
    data=data,
    loss_fn="cross_entropy"
)

# Optimizer step
hf_backend.optim_step(
    model_id="my-model",
    adam_params=hf_backend.AdamParams(learning_rate=5e-4)
)
```

## Custom Loss Functions 🎯

Define custom loss functions and upload them to the server with a simple decorator:

```python
from tinker_client import TinkerClient, custom_loss, LossFnOutput
import torch

client = TinkerClient("http://localhost:8000")

@custom_loss(client, name="my_dpo_loss")
def dpo_loss(model_outputs, loss_fn_inputs, attention_mask, beta=0.1):
    """Your custom DPO implementation"""
    # Your loss logic here
    ...
    return LossFnOutput(loss=loss, logprobs=logprobs, diagnostics={})

# Now use it in training!
result = hf_backend.forward_backward(
    model_id="my-model",
    data=data,
    loss_fn="my_dpo_loss",
    loss_fn_inputs={...}
)
```

**See [CUSTOM_LOSS_GUIDE.md](CUSTOM_LOSS_GUIDE.md) for complete documentation.**

## Built-in Loss Functions

- **cross_entropy**: Standard supervised learning (negative log-likelihood)
- **importance_sampling**: REINFORCE with importance sampling for RL
- **ppo**: Proximal Policy Optimization with clipping

## Architecture

```
┌─────────────────┐
│  Client Code    │  (Your training scripts with @custom_loss decorators)
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  Flask Server   │  (app.py - REST API endpoints)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Worker Thread  │  (worker.py - Job queue processing)
└────────┬────────┘
         │
         ├──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼
    ┌────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
    │   HF   │ │ Megatron│ │  vLLM  │ │ Loss   │
    │Backend │ │ Backend │ │Backend │ │Registry│
    └────────┘ └─────────┘ └────────┘ └────────┘
```

## Project Structure

```
Tinkerbell/
├── src/
│   ├── app.py                  # Flask server (REST API)
│   ├── worker.py               # Worker thread for job processing
│   ├── loss_functions.py       # Built-in and custom loss functions
│   ├── hf_backend.py           # HuggingFace + PEFT backend
│   ├── megatron_backend.py     # Megatron-LM backend (optional)
│   ├── vllm_backend.py         # vLLM backend (optional)
│   └── storage.py              # Storage utilities
├── examples/
│   ├── example_simple.py                    # Basic training example
│   ├── example_custom_loss_decorator.py     # Custom loss examples
│   └── example_simple_custom_loss.py        # Minimal custom loss
├── tinker_client.py            # Client library with @custom_loss decorator
├── CUSTOM_LOSS_GUIDE.md        # Complete guide to custom losses
└── requirements.txt            # Python dependencies
```

## Examples

See the `examples/` directory for complete examples:

- **example_simple.py**: Basic supervised learning
- **example_tinker_client.py**: Using Tinker SDK
- **example_tinker_full.py**: Complete workflow with LoRA
- **example_custom_loss_decorator.py**: Custom loss functions (DPO, contrastive, KL-regularized)
- **example_simple_custom_loss.py**: Minimal custom loss example

## Environment Variables

- `PORT`: Server port (default: 8000)
- `USE_MEGATRON`: Enable Megatron backend (default: false)
- `USE_VLLM`: Enable vLLM backend (default: false)
- `MAX_WORKERS`: Thread pool size for concurrent training (default: 4)

## API Endpoints

- `GET /healthz` - Health check
- `GET /get_server_capabilities` - List supported models
- `POST /fwdbwd` - Forward-backward pass
- `POST /optim_step` - Optimizer step
- `POST /register_custom_loss` - Upload custom loss function
- `GET /list_loss_functions` - List all available loss functions
- `POST /add_lora` - Create LoRA adapter
- `POST /remove_lora` - Delete LoRA adapter
- `POST /api/v1/sample` - Generate text (synchronous)
- `POST /api/v1/asample` - Generate text (asynchronous)
- `POST /retrieve_future` - Get async job result

## Advanced Features

### LoRA Multi-Adapter Training

Train multiple LoRA adapters concurrently:

```python
# Create multiple adapters
hf_backend.create_lora_adapter("model-a", lora_config)
hf_backend.create_lora_adapter("model-b", lora_config)

# Train them independently
result_a = hf_backend.forward_backward(model_id="model-a", data=data_a, loss_fn="cross_entropy")
result_b = hf_backend.forward_backward(model_id="model-b", data=data_b, loss_fn="cross_entropy")
```

### vLLM Co-located Sampling

Enable high-performance sampling with vLLM:

```bash
export USE_VLLM=true
python src/app.py
```

LoRA adapters are automatically registered with vLLM for immediate sampling.

### Megatron-LM Backend

For large-scale training with model parallelism:

```bash
export USE_MEGATRON=true
python src/app.py
```

## Contributing

This is a toy implementation for educational purposes. Contributions are welcome!

## License

MIT

## Related Projects

- [Tinker by Thinking Machines](https://tinker-docs.thinkingmachines.ai/)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [vLLM](https://github.com/vllm-project/vllm)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
