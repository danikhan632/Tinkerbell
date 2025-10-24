"""
HuggingFace + PEFT backend for LoRA fine-tuning.

This module provides the core training logic using:
- HuggingFace Transformers for the base model
- PEFT for LoRA adapters
- Custom loss functions (cross_entropy, importance_sampling, ppo)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import threading
from contextlib import contextmanager

import loss_functions


# Global state
base_model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
lora_adapters: Dict[str, PeftModel] = {}
optimizers: Dict[str, torch.optim.Optimizer] = {}
gradients_accumulated: Dict[str, bool] = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thread safety locks
_base_model_lock = threading.RLock()  # For base model initialization
_adapters_lock = threading.RLock()     # For adapter dict operations
_adapter_locks: Dict[str, threading.RLock] = {}  # Per-adapter locks for training/inference


@contextmanager
def _get_adapter_lock(model_id: str):
    """Get or create a lock for a specific adapter."""
    with _adapters_lock:
        if model_id not in _adapter_locks:
            _adapter_locks[model_id] = threading.RLock()
        lock = _adapter_locks[model_id]

    with lock:
        yield


@dataclass
class LoraConfigParams:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class AdamParams:
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


class ChatDataset(Dataset):
    def __init__(self, data: List[List[Dict[str, str]]], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[idx]

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        else:
            # Fallback
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def initialize_base_model(model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
    """Initialize the base model and tokenizer (thread-safe)."""
    global base_model, tokenizer

    with _base_model_lock:
        # Double-check pattern to avoid reloading
        if base_model is not None:
            return base_model, tokenizer

        print(f"Loading base model from {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        base_model.eval()

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        print(f"Base model loaded on {device}")
        return base_model, tokenizer


def create_lora_adapter(model_id: str, lora_config: Optional[LoraConfigParams] = None) -> PeftModel:
    """Create a new LoRA adapter for the base model (thread-safe)."""
    if base_model is None:
        initialize_base_model()

    with _adapters_lock:
        # Check if adapter already exists
        if model_id in lora_adapters:
            print(f"LoRA adapter '{model_id}' already exists, returning existing adapter")
            return lora_adapters[model_id]

        if lora_config is None:
            lora_config = LoraConfigParams()

        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )

        adapter = get_peft_model(base_model, peft_config)
        adapter.print_trainable_parameters()

        lora_adapters[model_id] = adapter
        gradients_accumulated[model_id] = False

        print(f"Created LoRA adapter '{model_id}'")
        return adapter


def forward_backward(
    model_id: str,
    data: List[List[Dict[str, str]]],
    loss_fn: str = "cross_entropy",
    loss_fn_inputs: Optional[Dict[str, Any]] = None,
    lora_config: Optional[LoraConfigParams] = None
) -> Dict[str, Any]:
    """
    Perform forward-backward pass with loss computation (thread-safe per adapter).

    Args:
        model_id: ID of the LoRA adapter
        data: Training data as list of conversations
        loss_fn: Loss function name ("cross_entropy", "importance_sampling", "ppo")
        loss_fn_inputs: Additional inputs for RL losses (target_tokens, logprobs, advantages)
        lora_config: LoRA configuration (if creating new adapter)

    Returns:
        Dict with loss, metrics, and metadata
    """
    global lora_adapters, gradients_accumulated

    # Create adapter if it doesn't exist (thread-safe)
    if model_id not in lora_adapters:
        adapter = create_lora_adapter(model_id, lora_config)
    else:
        with _adapters_lock:
            adapter = lora_adapters[model_id]

    # Use per-adapter lock to allow concurrent operations on different adapters
    with _get_adapter_lock(model_id):
        # Create dataset
        dataset = ChatDataset(data, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        adapter.train()
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
            outputs = adapter(
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
                model_outputs=outputs.logits,
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
            "backend": "huggingface",
            "metrics": avg_metrics,
            "loss_fn_outputs": {"loss_fn": loss_fn}
        }


def optim_step(model_id: str, adam_params: Optional[AdamParams] = None) -> Dict[str, Any]:
    """
    Apply optimizer step to a specific adapter (thread-safe per adapter).

    Args:
        model_id: ID of the LoRA adapter
        adam_params: Adam optimizer parameters

    Returns:
        Dict with metrics and metadata
    """
    with _adapters_lock:
        if model_id not in lora_adapters:
            raise ValueError(f"Adapter '{model_id}' not found")

        if not gradients_accumulated.get(model_id, False):
            raise ValueError(f"No gradients accumulated for '{model_id}'. Call forward_backward first.")

        adapter = lora_adapters[model_id]

    with _get_adapter_lock(model_id):
        if adam_params is None:
            adam_params = AdamParams()

        # Create or update optimizer
        if model_id not in optimizers:
            optimizers[model_id] = torch.optim.Adam(
                adapter.parameters(),
                lr=adam_params.learning_rate,
                betas=(adam_params.beta1, adam_params.beta2),
                eps=adam_params.eps,
                weight_decay=adam_params.weight_decay
            )
        else:
            # Update optimizer hyperparameters
            optimizer = optimizers[model_id]
            for param_group in optimizer.param_groups:
                param_group['lr'] = adam_params.learning_rate
                param_group['betas'] = (adam_params.beta1, adam_params.beta2)
                param_group['eps'] = adam_params.eps
                param_group['weight_decay'] = adam_params.weight_decay

        # Perform optimizer step
        optimizer = optimizers[model_id]
        optimizer.step()
        optimizer.zero_grad()

        gradients_accumulated[model_id] = False

        print(f"[{model_id}] Optimizer step complete. LR: {adam_params.learning_rate}")

        return {
            "metrics": {
                "learning_rate": adam_params.learning_rate,
                "step_completed": 1.0
            },
            "model_id": model_id,
            "backend": "huggingface"
        }


def generate(
    model_id: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50
) -> Dict[str, Any]:
    """
    Generate text using a specific LoRA adapter (thread-safe per adapter).

    Args:
        model_id: ID of the LoRA adapter (or "base" for base model)
        messages: Chat messages
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter

    Returns:
        Dict with generated text
    """
    # Get model reference
    if model_id == "base":
        model = base_model
    else:
        with _adapters_lock:
            if model_id not in lora_adapters:
                raise ValueError(f"Model '{model_id}' not found")
            model = lora_adapters[model_id]

    # Use per-adapter lock for inference
    lock_context = _get_adapter_lock(model_id) if model_id != "base" else _base_model_lock

    with lock_context:
        model.eval()

        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant: "

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return {
            "generated_text": generated_text,
            "model_id": model_id
        }


def list_adapters() -> List[Dict[str, Any]]:
    """List all LoRA adapters."""
    adapters = []
    for model_id, adapter in lora_adapters.items():
        adapters.append({
            "model_id": model_id,
            "trainable_params": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
            "has_gradients": gradients_accumulated.get(model_id, False)
        })
    return adapters


def delete_adapter(model_id: str) -> bool:
    """Delete a LoRA adapter (thread-safe)."""
    with _adapters_lock:
        if model_id in lora_adapters:
            # Use adapter lock to ensure no operations are in progress
            with _get_adapter_lock(model_id):
                del lora_adapters[model_id]
                if model_id in optimizers:
                    del optimizers[model_id]
                if model_id in gradients_accumulated:
                    del gradients_accumulated[model_id]
                if model_id in _adapter_locks:
                    del _adapter_locks[model_id]
                print(f"Deleted adapter '{model_id}'")
                return True
        return False


def list_loss_functions() -> Dict[str, Any]:
    """List available loss functions."""
    available = loss_functions.LOSS_REGISTRY.list_available()
    descriptions = {
        "cross_entropy": "Standard supervised learning (negative log-likelihood)",
        "importance_sampling": "REINFORCE with importance sampling for RL",
        "ppo": "Proximal Policy Optimization with clipping for RL"
    }
    return {
        "available_loss_functions": available,
        "descriptions": descriptions
    }
