"""
Megatron + PEFT backend for distributed LoRA fine-tuning.

This module provides distributed training using:
- Megatron-LM for distributed model parallelism
- PEFT for LoRA adapters (with Megatron support)
- Per-user/adapter optimizers with independent learning rates
- Custom loss functions (cross_entropy, importance_sampling, ppo)
- Concurrent adapter support with sequential training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
from contextlib import contextmanager

# Try importing Megatron - it's optional but required for this backend
try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.training import get_tokenizer as get_megatron_tokenizer
    MEGATRON_AVAILABLE = True
except ImportError as e:
    print(f"Megatron-LM not available for megatron_backend: {e}")
    MEGATRON_AVAILABLE = False
    # Define dummy types for type hints
    parallel_state = None
    tensor_parallel = None
    TransformerConfig = None
    GPTModel = None
    get_megatron_tokenizer = None

try:
    from peft import get_peft_model, LoraConfig, PeftModel
    PEFT_AVAILABLE = True
except ImportError as e:
    print(f"PEFT not available for megatron_backend: {e}")
    PEFT_AVAILABLE = False
    get_peft_model = None
    LoraConfig = None
    PeftModel = None

import loss_functions


# Global state
base_model = None  # Optional[GPTModel]
megatron_config = None  # Optional[TransformerConfig]
tokenizer = None  # Megatron tokenizer
lora_adapters: Dict[str, PeftModel] = {}  # Maps adapter_id -> PeftModel reference
optimizers: Dict[str, torch.optim.Optimizer] = {}  # Per-user/adapter optimizers
gradients_accumulated: Dict[str, bool] = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thread safety locks
_base_model_lock = threading.RLock()  # For base model initialization
_adapters_lock = threading.RLock()     # For adapter dict operations
_training_lock = threading.RLock()     # Global lock for sequential training (PEFT limitation)

# Shared PEFT model (all adapters use the same PeftModel instance)
peft_model: Optional[PeftModel] = None


@dataclass
class LoraConfigParams:
    """LoRA configuration parameters."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for GPT models
            self.target_modules = ["qkv_proj", "dense"]


@dataclass
class AdamParams:
    """Adam optimizer parameters."""
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


class ChatDataset(Dataset):
    """Dataset for chat-format training data."""

    def __init__(self, data: List[Dict[str, str]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx].get("messages", [])

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        }


def initialize_base_model(
    model: GPTModel,
    config: TransformerConfig,
) -> None:
    """
    Initialize the base Megatron model and tokenizer (thread-safe).

    Args:
        model: Megatron GPTModel instance
        config: Megatron TransformerConfig
    """
    global base_model, megatron_config, tokenizer

    with _base_model_lock:
        if base_model is not None:
            print("Base Megatron model already initialized")
            return

        base_model = model
        megatron_config = config

        # Get Megatron tokenizer
        if get_megatron_tokenizer is not None:
            tokenizer = get_megatron_tokenizer()
        else:
            raise RuntimeError("Megatron tokenizer not available")

        print(f"Base Megatron model initialized")
        print(f"  Model parallel size: {parallel_state.get_tensor_model_parallel_world_size()}")
        print(f"  Pipeline parallel size: {parallel_state.get_pipeline_model_parallel_world_size()}")


def create_lora_adapter(model_id: str, lora_config: Optional[LoraConfigParams] = None) -> PeftModel:
    """
    Create a new LoRA adapter for the base Megatron model (thread-safe).

    Uses PEFT's multi-adapter support. All adapters share the same PeftModel instance.

    Args:
        model_id: Unique identifier for the adapter
        lora_config: LoRA configuration parameters

    Returns:
        PeftModel instance (shared across all adapters)
    """
    if base_model is None or megatron_config is None:
        raise RuntimeError("Base Megatron model not initialized. Call initialize_base_model() first.")

    with _adapters_lock:
        # Check if adapter already exists
        if model_id in lora_adapters:
            print(f"LoRA adapter '{model_id}' already exists, returning existing adapter")
            return lora_adapters[model_id]

        if lora_config is None:
            lora_config = LoraConfigParams()

        # Create PEFT config with Megatron support
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            # Megatron-specific config
            megatron_config=megatron_config,
            megatron_core="megatron.core",
        )

        # Use PEFT's multi-adapter support
        # All adapters share the same PeftModel instance, we just switch between them
        global peft_model

        if not lora_adapters:
            # First adapter - wrap base model with PEFT
            peft_model = get_peft_model(base_model, peft_config)
            peft_model.print_trainable_parameters()
        else:
            # Subsequent adapters - add to existing PEFT model
            peft_model.add_adapter(model_id, peft_config)
            print(f"Added adapter '{model_id}' to PEFT model")

        # Store reference to the shared PEFT model
        lora_adapters[model_id] = peft_model
        gradients_accumulated[model_id] = False

        print(f"Created LoRA adapter '{model_id}'")
        return peft_model


def _create_or_update_optimizer(model_id: str, adam_params: AdamParams) -> torch.optim.Optimizer:
    """
    Create or update optimizer for a specific adapter (internal helper).

    Each user/adapter gets their own optimizer instance to allow independent
    learning rates and optimization states.

    Args:
        model_id: ID of the LoRA adapter
        adam_params: Adam optimizer parameters

    Returns:
        Optimizer instance for this adapter
    """
    global peft_model

    if model_id not in optimizers:
        # Create new optimizer for this adapter
        # Get only the parameters for this specific adapter
        adapter_params = [p for n, p in peft_model.named_parameters() if model_id in n and p.requires_grad]

        if not adapter_params:
            raise ValueError(f"No trainable parameters found for adapter '{model_id}'")

        optimizers[model_id] = torch.optim.Adam(
            adapter_params,
            lr=adam_params.learning_rate,
            betas=(adam_params.beta1, adam_params.beta2),
            eps=adam_params.eps,
            weight_decay=adam_params.weight_decay
        )
        print(f"[{model_id}] Created new optimizer with LR={adam_params.learning_rate}")
    else:
        # Update existing optimizer hyperparameters
        opt = optimizers[model_id]
        for param_group in opt.param_groups:
            param_group['lr'] = adam_params.learning_rate
            param_group['betas'] = (adam_params.beta1, adam_params.beta2)
            param_group['eps'] = adam_params.eps
            param_group['weight_decay'] = adam_params.weight_decay
        print(f"[{model_id}] Updated optimizer with LR={adam_params.learning_rate}")

    return optimizers[model_id]


def forward_backward(
    model_id: str,
    data: List[Dict[str, str]],
    loss_fn: str = "cross_entropy",
    loss_fn_inputs: Optional[Dict[str, Any]] = None,
    lora_config: Optional[LoraConfigParams] = None
) -> Dict[str, Any]:
    """
    Perform forward-backward pass for a specific adapter.

    Uses global training lock to ensure sequential processing (PEFT limitation).
    Multiple adapters can be created, but training is sequential.

    Args:
        model_id: ID of the LoRA adapter
        data: List of training samples in chat format
        loss_fn: Loss function name (cross_entropy, importance_sampling, ppo)
        loss_fn_inputs: Additional inputs for RL losses (target_tokens, logprobs, advantages)
        lora_config: LoRA configuration (if creating new adapter)

    Returns:
        Dict with loss, metrics, and metadata
    """
    global peft_model, gradients_accumulated

    # Create adapter if it doesn't exist (thread-safe)
    if model_id not in lora_adapters:
        create_lora_adapter(model_id, lora_config)

    # IMPORTANT: Use global training lock for sequential processing
    # PEFT's multi-adapter design has a single active_adapter state,
    # so concurrent training would interfere. We process sequentially.
    with _training_lock:
        # Get the shared PEFT model and switch to this adapter
        if peft_model is None:
            raise RuntimeError(f"Adapter '{model_id}' not found and PEFT model not initialized")

        # Switch to the correct adapter
        peft_model.set_adapter(model_id)
        print(f"[{model_id}] Switched to adapter, starting forward-backward pass")

        # Create dataset
        dataset = ChatDataset(data, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        peft_model.train()
        total_loss = 0.0
        num_batches = 0
        all_metrics = {}

        # Prepare loss function inputs
        loss_fn_inputs_dict = loss_fn_inputs or {}

        # Training loop
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass to get logits
            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Prepare loss function inputs for this batch
            batch_loss_fn_inputs = {}
            if loss_fn == "cross_entropy":
                # For supervised learning, use labels as target tokens
                batch_loss_fn_inputs["target_tokens"] = batch.get("labels", input_ids).to(device)
                batch_loss_fn_inputs["weights"] = (batch_loss_fn_inputs["target_tokens"] != -100).float()
            else:
                # For RL losses, extract from loss_fn_inputs
                seq_len = input_ids.shape[1]
                for key, values_list in loss_fn_inputs_dict.items():
                    if batch_idx < len(values_list):
                        # Convert to tensor and add batch dimension [batch_size=1, seq_len]
                        tensor_value = torch.tensor(values_list[batch_idx]).unsqueeze(0)

                        # Pad to match sequence length
                        current_len = tensor_value.shape[1]
                        if current_len < seq_len:
                            # Pad with appropriate values
                            if key == "target_tokens":
                                # Pad tokens with -100 (ignore index)
                                pad_value = -100
                                tensor_value = tensor_value.long()
                            else:
                                # Pad logprobs/advantages with 0
                                pad_value = 0.0
                                tensor_value = tensor_value.float()

                            padding = torch.full((1, seq_len - current_len), pad_value, dtype=tensor_value.dtype)
                            tensor_value = torch.cat([tensor_value, padding], dim=1)

                        batch_loss_fn_inputs[key] = tensor_value.to(device)

            # Compute loss using loss function module
            loss_output = loss_functions.compute_loss(
                loss_fn_name=loss_fn,
                model_outputs=outputs.logits if hasattr(outputs, 'logits') else outputs,
                loss_fn_inputs=batch_loss_fn_inputs,
                attention_mask=attention_mask
            )

            # Backward pass (gradients accumulate)
            loss_output.loss.backward()

            total_loss += loss_output.loss.item()
            num_batches += 1

            # Accumulate metrics
            for key, value in loss_output.diagnostics.items():
                if key not in all_metrics:
                    all_metrics[key] = 0.0
                all_metrics[key] += value

        gradients_accumulated[model_id] = True
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Average metrics
        avg_metrics = {key: value / num_batches for key, value in all_metrics.items()} if num_batches > 0 else {}

        print(f"[{model_id}] Forward-backward complete. Loss: {avg_loss:.4f}, Samples: {num_batches}")

        return {
            "loss": avg_loss,
            "num_samples": num_batches,
            "model_id": model_id,
            "backend": "megatron",
            "metrics": avg_metrics,
            "loss_fn_outputs": {"loss_fn": loss_fn}
        }


def optim_step(model_id: str, adam_params: Optional[AdamParams] = None) -> Dict[str, Any]:
    """
    Apply optimizer step to a specific adapter.

    Each adapter has its own optimizer with independent learning rate
    and optimization state, enabling per-user customization.

    Uses global training lock to ensure sequential processing.

    Args:
        model_id: ID of the LoRA adapter (user identifier)
        adam_params: Adam optimizer parameters

    Returns:
        Dict with metrics and metadata
    """
    with _adapters_lock:
        if model_id not in lora_adapters:
            raise ValueError(f"Adapter '{model_id}' not found")

        if not gradients_accumulated.get(model_id, False):
            raise ValueError(f"No gradients accumulated for '{model_id}'. Call forward_backward first.")

    with _training_lock:
        global peft_model

        # Switch to the correct adapter
        peft_model.set_adapter(model_id)

        if adam_params is None:
            adam_params = AdamParams()

        # Create or update optimizer for this specific user/adapter
        opt = _create_or_update_optimizer(model_id, adam_params)

        # Perform optimizer step
        opt.step()
        opt.zero_grad()

        gradients_accumulated[model_id] = False

        print(f"[{model_id}] Optimizer step complete. LR: {adam_params.learning_rate}")

        return {
            "metrics": {
                "learning_rate": adam_params.learning_rate,
                "step_completed": 1.0
            },
            "model_id": model_id,
            "backend": "megatron"
        }


def get_optimizer_state(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current optimizer state for a specific user/adapter.

    Args:
        model_id: Adapter ID

    Returns:
        Dict with optimizer state information or None if not found
    """
    with _adapters_lock:
        if model_id not in optimizers:
            return None

        opt = optimizers[model_id]
        if not opt.param_groups:
            return None

        # Return state from first param group (all groups have same hyperparams)
        param_group = opt.param_groups[0]
        return {
            "model_id": model_id,
            "learning_rate": param_group['lr'],
            "beta1": param_group['betas'][0],
            "beta2": param_group['betas'][1],
            "eps": param_group['eps'],
            "weight_decay": param_group['weight_decay'],
            "num_params": sum(p.numel() for p in param_group['params']),
        }


def list_optimizer_states() -> List[Dict[str, Any]]:
    """
    List optimizer states for all users/adapters.

    Returns:
        List of optimizer state dicts
    """
    with _adapters_lock:
        states = []
        for model_id in optimizers.keys():
            state = get_optimizer_state(model_id)
            if state:
                states.append(state)
        return states


def remove_lora_adapter(model_id: str) -> bool:
    """
    Remove a LoRA adapter and its associated optimizer.

    Args:
        model_id: Adapter ID to remove

    Returns:
        True if adapter was removed, False if not found
    """
    with _adapters_lock:
        if model_id in lora_adapters:
            # Note: PEFT doesn't have a remove_adapter method yet
            # So we just remove from our tracking dict
            del lora_adapters[model_id]
            if model_id in optimizers:
                del optimizers[model_id]
                print(f"Removed optimizer for '{model_id}'")
            if model_id in gradients_accumulated:
                del gradients_accumulated[model_id]

            print(f"Removed LoRA adapter '{model_id}' from tracking")
            return True
        return False


def list_lora_adapters() -> List[Dict[str, str]]:
    """
    List all registered LoRA adapters.

    Returns:
        List of dicts with adapter_id
    """
    with _adapters_lock:
        return [{"adapter_id": aid} for aid in lora_adapters.keys()]


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the Megatron backend.

    Returns:
        Dict with backend configuration and status
    """
    global peft_model

    if peft_model is None:
        return {
            "initialized": False,
            "active_adapters": 0,
            "mode": "not_initialized",
            "backend": "megatron"
        }

    with _adapters_lock:
        return {
            "initialized": True,
            "active_adapters": len(lora_adapters),
            "adapter_ids": list(lora_adapters.keys()),
            "mode": "distributed (Megatron + PEFT)",
            "backend": "megatron",
            "tensor_parallel_size": parallel_state.get_tensor_model_parallel_world_size(),
            "pipeline_parallel_size": parallel_state.get_pipeline_model_parallel_world_size(),
        }
