"""
Example: Custom Loss Functions with Decorator

This example shows how to define custom loss functions using the @custom_loss decorator,
which automatically uploads them to the Tinkerbell server.

Usage:
    1. Start the server: python src/app.py
    2. Run this example: python examples/example_custom_loss_decorator.py
"""

import sys
import os

# Add parent directory to path to import tinker_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tinker_client import TinkerClient, custom_loss, LossFnOutput
import torch
from typing import Dict
import requests
import time


# Initialize client
client = TinkerClient(base_url="http://localhost:8000")


# ============================================================================
# Example 1: DPO (Direct Preference Optimization) Loss
# ============================================================================

@custom_loss(client, name="dpo")
def dpo_loss(
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    beta: float = 0.1
):
    """
    Direct Preference Optimization loss for preference learning.

    This loss is used in RLHF to optimize a model based on pairwise preferences
    without requiring a separate reward model.

    Args:
        model_outputs: Model logits (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict with:
            - chosen_tokens: Preferred response tokens
            - rejected_tokens: Rejected response tokens
            - reference_chosen_logprobs: Reference model log probs for chosen
            - reference_rejected_logprobs: Reference model log probs for rejected
        attention_mask: Attention mask
        beta: DPO temperature parameter (default: 0.1)

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    chosen_tokens = loss_fn_inputs["chosen_tokens"]
    rejected_tokens = loss_fn_inputs["rejected_tokens"]
    ref_chosen_logprobs = loss_fn_inputs["reference_chosen_logprobs"]
    ref_rejected_logprobs = loss_fn_inputs["reference_rejected_logprobs"]

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Get log probs for chosen and rejected
    chosen_tokens_safe = chosen_tokens.clone()
    chosen_tokens_safe[chosen_tokens == -100] = 0
    policy_chosen_logprobs = log_probs.gather(
        dim=-1, index=chosen_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    rejected_tokens_safe = rejected_tokens.clone()
    rejected_tokens_safe[rejected_tokens == -100] = 0
    policy_rejected_logprobs = log_probs.gather(
        dim=-1, index=rejected_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Apply attention mask
    policy_chosen_logprobs = policy_chosen_logprobs * attention_mask
    policy_rejected_logprobs = policy_rejected_logprobs * attention_mask

    # Compute scores
    num_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    policy_chosen_score = policy_chosen_logprobs.sum(dim=1) / num_tokens.squeeze()
    policy_rejected_score = policy_rejected_logprobs.sum(dim=1) / num_tokens.squeeze()
    ref_chosen_score = ref_chosen_logprobs.sum(dim=1) / num_tokens.squeeze()
    ref_rejected_score = ref_rejected_logprobs.sum(dim=1) / num_tokens.squeeze()

    # DPO loss
    policy_diff = policy_chosen_score - policy_rejected_score
    ref_diff = ref_chosen_score - ref_rejected_score
    logits = beta * (policy_diff - ref_diff)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # Metrics
    reward_accuracy = (policy_diff > 0).float().mean()

    return LossFnOutput(
        loss=loss,
        logprobs=policy_chosen_logprobs,
        diagnostics={
            "loss:sum": loss.item(),
            "reward_accuracy": reward_accuracy.item(),
            "mean_policy_diff": policy_diff.mean().item(),
        }
    )


# ============================================================================
# Example 2: Contrastive Loss
# ============================================================================

@custom_loss(client, name="contrastive")
def contrastive_loss(
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    temperature: float = 0.07
):
    """
    Contrastive learning loss for distinguishing positive from negative examples.

    Args:
        model_outputs: Model logits
        loss_fn_inputs: Dict with positive_tokens and negative_tokens
        attention_mask: Attention mask
        temperature: Temperature parameter for contrastive learning

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    positive_tokens = loss_fn_inputs["positive_tokens"]
    negative_tokens = loss_fn_inputs["negative_tokens"]

    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Get log probs for positive and negative
    pos_tokens_safe = positive_tokens.clone()
    pos_tokens_safe[positive_tokens == -100] = 0
    pos_logprobs = log_probs.gather(
        dim=-1, index=pos_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    neg_tokens_safe = negative_tokens.clone()
    neg_tokens_safe[negative_tokens == -100] = 0
    neg_logprobs = log_probs.gather(
        dim=-1, index=neg_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Apply mask and compute scores
    pos_logprobs = pos_logprobs * attention_mask
    neg_logprobs = neg_logprobs * attention_mask

    num_tokens = attention_mask.sum().clamp(min=1)
    pos_score = pos_logprobs.sum() / num_tokens
    neg_score = neg_logprobs.sum() / num_tokens

    # Contrastive loss
    logits = torch.stack([pos_score, neg_score]) / temperature
    labels = torch.tensor([0], device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), labels)

    return LossFnOutput(
        loss=loss,
        logprobs=pos_logprobs,
        diagnostics={
            "loss:sum": loss.item(),
            "positive_score": pos_score.item(),
            "negative_score": neg_score.item(),
            "score_gap": (pos_score - neg_score).item(),
        }
    )


# ============================================================================
# Example 3: KL-Divergence Regularized Loss
# ============================================================================

@custom_loss(client, name="kl_regularized")
def kl_regularized_loss(
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    kl_weight: float = 0.1
):
    """
    Cross-entropy loss with KL divergence regularization against a reference model.

    Useful for preventing the model from diverging too far from a reference policy.

    Args:
        model_outputs: Model logits
        loss_fn_inputs: Dict with target_tokens, weights, and reference_logprobs
        attention_mask: Attention mask
        kl_weight: Weight for KL divergence penalty

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    target_tokens = loss_fn_inputs["target_tokens"]
    weights = loss_fn_inputs.get("weights")
    reference_logprobs = loss_fn_inputs.get("reference_logprobs")

    if weights is None:
        weights = torch.ones_like(target_tokens, dtype=torch.float32)

    # Standard cross-entropy loss
    target_tokens_safe = target_tokens.clone()
    target_tokens_safe[target_tokens == -100] = 0

    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)
    target_logprobs = log_probs.gather(
        dim=-1, index=target_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    ce_loss = (-target_logprobs * weights * attention_mask).sum()

    # KL divergence regularization (if reference provided)
    kl_loss = 0.0
    if reference_logprobs is not None:
        # KL(policy || reference) ≈ Σ policy * (log(policy) - log(reference))
        probs = torch.exp(log_probs)
        kl_per_token = (probs * (log_probs - reference_logprobs.unsqueeze(-1))).sum(dim=-1)
        kl_loss = (kl_per_token * attention_mask).sum()

    # Combined loss
    total_loss = ce_loss + kl_weight * kl_loss

    num_tokens = (weights * attention_mask).sum().clamp(min=1)
    mean_nll = ce_loss.item() / num_tokens.item()

    return LossFnOutput(
        loss=total_loss,
        logprobs=target_logprobs,
        diagnostics={
            "loss:sum": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss if isinstance(kl_loss, float) else kl_loss.item(),
            "mean_nll": mean_nll,
        }
    )


# ============================================================================
# Main: Demonstrate usage
# ============================================================================

def main():
    print("=== Custom Loss Function Decorator Example ===\n")

    # Check server health
    try:
        health = client.healthz()
        print(f"✓ Connected to server: {health}\n")
    except Exception as e:
        print(f"✗ Could not connect to server: {e}")
        print("Make sure the server is running: python src/app.py")
        return

    # The decorators above already registered the loss functions!
    print("Custom loss functions have been registered via @custom_loss decorator:\n")
    print("  1. dpo - Direct Preference Optimization")
    print("  2. contrastive - Contrastive learning loss")
    print("  3. kl_regularized - CE with KL regularization\n")

    # List all available loss functions
    print("Querying server for all available loss functions...")
    try:
        losses = client.list_loss_functions()
        print(f"\nAvailable loss functions on server:")
        for loss_name in losses.get("available_loss_functions", []):
            desc = losses.get("descriptions", {}).get(loss_name, "Custom loss function")
            print(f"  - {loss_name}: {desc}")
    except Exception as e:
        print(f"Could not query loss functions: {e}")

    print("\n" + "="*60)
    print("\nNow you can use these custom losses in training!")
    print("\nExample:")
    print("""
    # In your training code:
    result = forward_backward(
        model_id="my-model",
        data=training_data,
        loss_fn="dpo",  # Use your custom loss!
        loss_fn_inputs={
            "chosen_tokens": [...],
            "rejected_tokens": [...],
            "reference_chosen_logprobs": [...],
            "reference_rejected_logprobs": [...]
        }
    )
    """)

    print("\n" + "="*60)
    print("\nKey Features:")
    print("✓ Define loss functions with standard Python code")
    print("✓ Decorator automatically uploads to server")
    print("✓ Functions work with PyTorch tensors")
    print("✓ Support for custom hyperparameters")
    print("✓ Full control over diagnostics and metrics")


if __name__ == "__main__":
    main()
