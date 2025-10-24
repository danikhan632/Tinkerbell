"""
Loss functions for fine-tuning language models.

Implements Tinker-compatible loss functions:
- cross_entropy: Standard supervised learning
- importance_sampling: REINFORCE with importance sampling for RL
- ppo: Proximal Policy Optimization for RL
"""

import torch
from typing import Dict, Tuple, List, Any, Callable
from dataclasses import dataclass


@dataclass
class LossFnOutput:
    """Output from a loss function computation"""
    loss: torch.Tensor  # Scalar loss value
    logprobs: torch.Tensor  # Log probabilities for each token
    diagnostics: Dict[str, float]  # Additional metrics


class LossFunctionRegistry:
    """Registry for built-in loss functions"""

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._register_builtin_losses()

    def _register_builtin_losses(self):
        """Register all built-in loss functions"""
        self.register("cross_entropy", cross_entropy_loss)
        self.register("importance_sampling", importance_sampling_loss)
        self.register("ppo", ppo_loss)

    def register(self, name: str, loss_fn: Callable):
        """Register a loss function"""
        self._registry[name] = loss_fn

    def get(self, name: str) -> Callable:
        """Get a registered loss function"""
        if name not in self._registry:
            raise ValueError(
                f"Unknown loss function: {name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_available(self) -> List[str]:
        """List all available loss functions"""
        return list(self._registry.keys())


def cross_entropy_loss(
    model_outputs: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
) -> LossFnOutput:
    """
    Standard cross-entropy loss for supervised learning.

    Optimizes the policy p_θ to maximize log-probability of tokens:
    L(θ) = -E_x[log p_θ(x)]

    Args:
        model_outputs: Logits from model (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict containing:
            - target_tokens: (batch_size, seq_len) - Target token IDs
            - weights: (batch_size, seq_len) - Token-level loss weights (0 or 1)
        attention_mask: (batch_size, seq_len) - Attention mask

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    target_tokens = loss_fn_inputs.get("target_tokens")
    weights = loss_fn_inputs.get("weights")

    if target_tokens is None:
        raise ValueError("cross_entropy requires 'target_tokens' in loss_fn_inputs")

    # Default weights to all 1s if not provided
    if weights is None:
        weights = torch.ones_like(target_tokens, dtype=torch.float32)

    # Replace -100 (ignore index) with 0 to avoid index errors in gather
    # The weights mask will zero out the loss for these positions
    target_tokens_safe = target_tokens.clone()
    target_tokens_safe[target_tokens == -100] = 0

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Gather log probs for target tokens
    batch_size, seq_len = target_tokens_safe.shape
    target_logprobs = log_probs.gather(
        dim=-1,
        index=target_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)  # (batch_size, seq_len)

    # Apply weights and compute elementwise loss
    elementwise_loss = -target_logprobs * weights

    # Apply attention mask (ignore padding)
    elementwise_loss = elementwise_loss * attention_mask

    # Sum reduction to get total loss
    loss = elementwise_loss.sum()

    # Compute diagnostics
    num_tokens = attention_mask.sum().item()
    num_weighted_tokens = (weights * attention_mask).sum().item()
    mean_nll = loss.item() / max(num_weighted_tokens, 1)

    diagnostics = {
        "loss:sum": loss.item(),
        "mean_nll": mean_nll,
        "num_tokens": num_tokens,
        "num_weighted_tokens": num_weighted_tokens,
        "perplexity": torch.exp(torch.tensor(mean_nll)).item(),
    }

    return LossFnOutput(
        loss=loss,
        logprobs=target_logprobs,
        diagnostics=diagnostics
    )


def importance_sampling_loss(
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
) -> LossFnOutput:
    """
    Importance sampling REINFORCE for RL.

    Uses importance weighting to correct for off-policy learning:
    L_IS(θ) = E_{x~q}[p_θ(x)/q(x) * A(x)]

    Args:
        model_outputs: Logits from model (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict containing:
            - target_tokens: (batch_size, seq_len) - Sampled tokens
            - logprobs: (batch_size, seq_len) - Reference log probs from sampler q
            - advantages: (batch_size, seq_len) - Advantage values
        attention_mask: (batch_size, seq_len) - Attention mask

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    target_tokens = loss_fn_inputs.get("target_tokens")
    sampling_logprobs = loss_fn_inputs.get("logprobs")
    advantages = loss_fn_inputs.get("advantages")

    if target_tokens is None:
        raise ValueError("importance_sampling requires 'target_tokens'")
    if sampling_logprobs is None:
        raise ValueError("importance_sampling requires 'logprobs' (from sampler)")
    if advantages is None:
        raise ValueError("importance_sampling requires 'advantages'")

    # Replace -100 (ignore index) with 0 to avoid index errors
    target_tokens_safe = target_tokens.clone()
    target_tokens_safe[target_tokens == -100] = 0

    # Compute current policy log probabilities
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)
    target_logprobs = log_probs.gather(
        dim=-1,
        index=target_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Compute importance ratio: p_θ(x) / q(x)
    log_ratio = target_logprobs - sampling_logprobs
    prob_ratio = torch.exp(log_ratio)

    # Clip ratio to prevent extreme importance weights (optional but recommended)
    prob_ratio = torch.clamp(prob_ratio, 0.01, 100.0)

    # Importance-weighted policy gradient
    weighted_advantages = prob_ratio * advantages

    # Apply attention mask
    weighted_advantages = weighted_advantages * attention_mask

    # Loss is negative of the objective (we minimize loss)
    loss = -weighted_advantages.sum()

    # Compute diagnostics
    num_tokens = attention_mask.sum().item()
    mean_advantage = (advantages * attention_mask).sum().item() / max(num_tokens, 1)
    mean_ratio = (prob_ratio * attention_mask).sum().item() / max(num_tokens, 1)

    diagnostics = {
        "loss:sum": loss.item(),
        "mean_advantage": mean_advantage,
        "mean_importance_ratio": mean_ratio,
        "max_importance_ratio": prob_ratio.max().item(),
        "min_importance_ratio": prob_ratio.min().item(),
        "num_tokens": num_tokens,
    }

    return LossFnOutput(
        loss=loss,
        logprobs=target_logprobs,
        diagnostics=diagnostics
    )


def ppo_loss(
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    clip_low: float = 0.8,  # 1 - epsilon_low (epsilon=0.2)
    clip_high: float = 1.2,  # 1 + epsilon_high (epsilon=0.2)
) -> LossFnOutput:
    """
    Proximal Policy Optimization (PPO) loss with clipping.

    Clips the importance ratio to prevent large policy updates:
    L_PPO(θ) = -E_{x~q}[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
    where r(θ) = p_θ(x)/q(x)

    Args:
        model_outputs: Logits from model (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict containing:
            - target_tokens: (batch_size, seq_len) - Sampled tokens
            - logprobs: (batch_size, seq_len) - Reference log probs from sampler q
            - advantages: (batch_size, seq_len) - Advantage values
        attention_mask: (batch_size, seq_len) - Attention mask
        clip_low: Lower clip threshold (default 0.8 = 1 - 0.2)
        clip_high: Upper clip threshold (default 1.2 = 1 + 0.2)

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    target_tokens = loss_fn_inputs.get("target_tokens")
    sampling_logprobs = loss_fn_inputs.get("logprobs")
    advantages = loss_fn_inputs.get("advantages")

    if target_tokens is None:
        raise ValueError("ppo requires 'target_tokens'")
    if sampling_logprobs is None:
        raise ValueError("ppo requires 'logprobs' (from sampler)")
    if advantages is None:
        raise ValueError("ppo requires 'advantages'")

    # Replace -100 (ignore index) with 0 to avoid index errors
    target_tokens_safe = target_tokens.clone()
    target_tokens_safe[target_tokens == -100] = 0

    # Compute current policy log probabilities
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)
    target_logprobs = log_probs.gather(
        dim=-1,
        index=target_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Compute importance ratio: p_θ(x) / q(x)
    log_ratio = target_logprobs - sampling_logprobs
    prob_ratio = torch.exp(log_ratio)

    # Clip the ratio
    clipped_ratio = torch.clamp(prob_ratio, clip_low, clip_high)

    # Compute both unclipped and clipped objectives
    unclipped_objective = prob_ratio * advantages
    clipped_objective = clipped_ratio * advantages

    # PPO objective: take minimum (most conservative)
    ppo_objective = torch.min(unclipped_objective, clipped_objective)

    # Apply attention mask
    ppo_objective = ppo_objective * attention_mask

    # Loss is negative of objective
    loss = -ppo_objective.sum()

    # Compute diagnostics
    num_tokens = attention_mask.sum().item()
    mean_advantage = (advantages * attention_mask).sum().item() / max(num_tokens, 1)
    mean_ratio = (prob_ratio * attention_mask).sum().item() / max(num_tokens, 1)

    # Compute fraction of clipped ratios
    clipped = ((prob_ratio < clip_low) | (prob_ratio > clip_high)).float()
    clip_fraction = (clipped * attention_mask).sum().item() / max(num_tokens, 1)

    diagnostics = {
        "loss:sum": loss.item(),
        "mean_advantage": mean_advantage,
        "mean_ratio": mean_ratio,
        "clip_fraction": clip_fraction,
        "max_ratio": prob_ratio.max().item(),
        "min_ratio": prob_ratio.min().item(),
        "num_tokens": num_tokens,
    }

    return LossFnOutput(
        loss=loss,
        logprobs=target_logprobs,
        diagnostics=diagnostics
    )


# Global registry instance (instantiated after loss functions are defined)
LOSS_REGISTRY = LossFunctionRegistry()


def compute_loss(
    loss_fn_name: str,
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, Any],
    attention_mask: torch.Tensor,
    **kwargs
) -> LossFnOutput:
    """
    Compute loss using a registered loss function.

    Args:
        loss_fn_name: Name of the loss function ("cross_entropy", "ppo", etc.)
        model_outputs: Model logits (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict of tensors required by the loss function
        attention_mask: Attention mask (batch_size, seq_len)
        **kwargs: Additional arguments passed to the loss function

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    # Convert numpy arrays to torch tensors if needed
    loss_fn_inputs_torch = {}
    for key, value in loss_fn_inputs.items():
        if isinstance(value, torch.Tensor):
            loss_fn_inputs_torch[key] = value
        else:
            # Assume numpy array or list
            import numpy as np
            if isinstance(value, np.ndarray):
                loss_fn_inputs_torch[key] = torch.from_numpy(value)
            else:
                loss_fn_inputs_torch[key] = torch.tensor(value)

    # Get loss function
    loss_fn = LOSS_REGISTRY.get(loss_fn_name)

    # Compute loss
    return loss_fn(model_outputs, loss_fn_inputs_torch, attention_mask, **kwargs)


def register_custom_loss(name: str, loss_fn: Callable):
    """
    Register a custom loss function.

    Example:
        def my_custom_loss(model_outputs, loss_fn_inputs, attention_mask):
            # Your custom loss logic
            ...
            return LossFnOutput(loss=loss, logprobs=logprobs, diagnostics={})

        register_custom_loss("my_loss", my_custom_loss)
    """
    LOSS_REGISTRY.register(name, loss_fn)
