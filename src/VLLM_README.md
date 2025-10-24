# vLLM Integration with LoRA Adapters

This integration adds high-performance inference using vLLM with LoRA adapter support to the Tink worker system.

## Features

- **High-performance sampling**: Uses vLLM for fast inference
- **LoRA adapter support**: Load and use multiple LoRA adapters for generation
- **Megatron compatible**: Works alongside Megatron for training
- **Thread-safe**: Supports concurrent requests with different adapters

## Setup

### 1. Install vLLM and dependencies

```bash
pip install vllm requests
# Or install TRL with vLLM support
pip install trl[vllm]
```

### 2. Start vLLM server

Start the vLLM server with LoRA support:

```bash
# Using TRL's vllm-serve command
trl vllm-serve \
    --model meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --port 8000

# Or using vLLM directly
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64 \
    --port 8000
```

### 3. Configure environment variables

Set these environment variables before starting the Tink worker:

```bash
# Enable vLLM backend
export USE_VLLM=true

# vLLM server connection (optional, defaults shown)
export VLLM_BASE_URL="http://localhost:8000"
# OR
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="8000"

# vLLM configuration
export VLLM_GROUP_PORT="51216"  # For weight updates
export VLLM_TIMEOUT="5.0"        # Connection timeout in seconds
export VLLM_ENABLE_LORA="true"   # Enable LoRA support
export VLLM_MAX_LORAS="4"        # Max concurrent LoRA adapters
export VLLM_MAX_LORA_RANK="64"   # Max LoRA rank

# Optionally enable Megatron for training
export USE_MEGATRON=true
export MEGATRON_SERVER_URL="http://localhost:5000"
```

## Usage

### Basic Sampling

Send a sampling request to the Tink API:

```python
import requests

response = requests.post("http://localhost:8000/api/v1/sample", json={
    "prompts": ["Once upon a time"],
    "sampling_params": {
        "max_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40
    }
})

print(response.json())
```

### Sampling with LoRA Adapter

First, register a LoRA adapter:

```python
import vllm_backend

# Register a LoRA adapter
vllm_backend.register_lora_adapter(
    adapter_id="my_lora_adapter",
    lora_path="/path/to/lora/weights"
)
```

Then use it in sampling requests:

```python
response = requests.post("http://localhost:8000/api/v1/sample", json={
    "prompts": ["Once upon a time"],
    "model_id": "my_lora_adapter",  # Specify the LoRA adapter
    "sampling_params": {
        "max_tokens": 50,
        "temperature": 0.8
    }
})
```

### Managing LoRA Adapters

```python
import vllm_backend

# Register an adapter
vllm_backend.register_lora_adapter(
    adapter_id="adapter_1",
    lora_path="/path/to/lora/adapter_1"
)

# List all registered adapters
adapters = vllm_backend.list_lora_adapters()
print(adapters)
# [{'adapter_id': 'adapter_1', 'lora_path': '/path/to/lora/adapter_1'}]

# Unregister an adapter
vllm_backend.unregister_lora_adapter("adapter_1")
```

### Advanced: Weight Updates

Initialize communicator for updating model weights:

```python
import vllm_backend
import torch
from transformers import AutoModelForCausalLM

# Initialize communicator
vllm_backend.init_communicator(device="cuda")

# Load a model with updated weights
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# ... train the model ...

# Update vLLM server weights
vllm_backend.update_model_weights(model)

# Close communicator when done
vllm_backend.close_communicator()
```

## Architecture

### Backend Selection

The worker supports multiple backends:

1. **vLLM** (`USE_VLLM=true`): High-performance sampling with LoRA support
2. **Megatron** (`USE_MEGATRON=true`): Training and sampling via Megatron server
3. Both can be enabled simultaneously (vLLM for sampling, Megatron for training)

### Request Flow

```
Client Request → Flask API → Worker Queue → vLLM Backend → vLLM Server (with LoRA) → Response
```

## LoRA Adapter Format

LoRA adapters should be saved in the standard PEFT format:

```
/path/to/lora/adapter/
├── adapter_config.json
├── adapter_model.bin (or adapter_model.safetensors)
└── ...
```

You can create LoRA adapters using PEFT:

```python
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)

# Train the model...

# Save LoRA adapter
peft_model.save_pretrained("/path/to/lora/adapter")
```

## Troubleshooting

### vLLM Server Not Reachable

```
ConnectionError: The vLLM server can't be reached at http://localhost:8000
```

**Solution**: Make sure the vLLM server is running:
```bash
trl vllm-serve --model meta-llama/Llama-2-7b-hf --enable-lora
```

### LoRA Adapter Not Found

```
ValueError: LoRA path does not exist: /path/to/lora
```

**Solution**: Ensure the LoRA adapter path is correct and the adapter is properly saved.

### Import Error

```
ImportError: vLLM is not available
```

**Solution**: Install vLLM:
```bash
pip install vllm requests
```

## API Reference

See `vllm_backend.py` for the complete API:

- `initialize_vllm_client(config)`: Initialize vLLM client
- `generate_with_vllm(...)`: Generate completions with optional LoRA
- `register_lora_adapter(adapter_id, lora_path)`: Register a LoRA adapter
- `unregister_lora_adapter(adapter_id)`: Remove a LoRA adapter
- `list_lora_adapters()`: List all registered adapters
- `update_model_weights(model)`: Update vLLM server weights
- `init_communicator(device)`: Initialize weight update communicator
- `close_communicator()`: Close communicator
- `reset_prefix_cache()`: Reset vLLM prefix cache
- `get_client_info()`: Get client status and configuration
