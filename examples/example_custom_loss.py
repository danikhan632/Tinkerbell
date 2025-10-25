"""
Example: Using Custom Loss Functions in Tinkerbell

This demonstrates how to:
1. Define a custom loss function
2. Register it with the Tinkerbell backend
3. Use it in training

The example shows how to implement a DPO (Direct Preference Optimization) loss,
which is commonly used in RLHF for preference learning.
"""

import asyncio
import sys
import os

# Add src to path so we can import the backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from typing import Dict
import loss_functions
from loss_functions import LossFnOutput, register_custom_loss

# Import backend (will be initialized when server starts)
import hf_backend


def dpo_loss(
    model_outputs: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    beta: float = 0.1,  # DPO temperature parameter
) -> LossFnOutput:
    """
    Direct Preference Optimization (DPO) loss.

    DPO optimizes the policy to prefer chosen responses over rejected ones:
    L_DPO = -log σ(β * [log π_θ(y_w|x) - log π_θ(y_l|x)])

    where:
    - y_w = chosen (winning) response
    - y_l = rejected (losing) response
    - σ = sigmoid function
    - β = temperature parameter

    Args:
        model_outputs: Logits from model (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict containing:
            - chosen_tokens: (batch_size, seq_len) - Preferred completion token IDs
            - rejected_tokens: (batch_size, seq_len) - Rejected completion token IDs
            - reference_chosen_logprobs: (batch_size, seq_len) - Log probs from reference model for chosen
            - reference_rejected_logprobs: (batch_size, seq_len) - Log probs from reference model for rejected
        attention_mask: (batch_size, seq_len) - Attention mask
        beta: DPO temperature hyperparameter (default: 0.1)

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    chosen_tokens = loss_fn_inputs.get("chosen_tokens")
    rejected_tokens = loss_fn_inputs.get("rejected_tokens")
    ref_chosen_logprobs = loss_fn_inputs.get("reference_chosen_logprobs")
    ref_rejected_logprobs = loss_fn_inputs.get("reference_rejected_logprobs")

    if chosen_tokens is None or rejected_tokens is None:
        raise ValueError("dpo requires 'chosen_tokens' and 'rejected_tokens'")
    if ref_chosen_logprobs is None or ref_rejected_logprobs is None:
        raise ValueError("dpo requires 'reference_chosen_logprobs' and 'reference_rejected_logprobs'")

    # Compute log probabilities from current policy
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Gather log probs for chosen tokens
    chosen_tokens_safe = chosen_tokens.clone()
    chosen_tokens_safe[chosen_tokens == -100] = 0
    policy_chosen_logprobs = log_probs.gather(
        dim=-1,
        index=chosen_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Gather log probs for rejected tokens
    rejected_tokens_safe = rejected_tokens.clone()
    rejected_tokens_safe[rejected_tokens == -100] = 0
    policy_rejected_logprobs = log_probs.gather(
        dim=-1,
        index=rejected_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Apply attention mask
    policy_chosen_logprobs = policy_chosen_logprobs * attention_mask
    policy_rejected_logprobs = policy_rejected_logprobs * attention_mask
    ref_chosen_logprobs = ref_chosen_logprobs * attention_mask
    ref_rejected_logprobs = ref_rejected_logprobs * attention_mask

    # Compute log-likelihood ratios (policy vs reference)
    # Sum over sequence length to get per-example scores
    num_tokens = attention_mask.sum(dim=1, keepdim=True)

    policy_chosen_score = policy_chosen_logprobs.sum(dim=1) / num_tokens.squeeze()
    policy_rejected_score = policy_rejected_logprobs.sum(dim=1) / num_tokens.squeeze()
    ref_chosen_score = ref_chosen_logprobs.sum(dim=1) / num_tokens.squeeze()
    ref_rejected_score = ref_rejected_logprobs.sum(dim=1) / num_tokens.squeeze()

    # DPO loss: -log sigmoid(beta * [policy_diff - ref_diff])
    policy_diff = policy_chosen_score - policy_rejected_score
    ref_diff = ref_chosen_score - ref_rejected_score

    logits = beta * (policy_diff - ref_diff)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # Compute reward accuracy (how often policy prefers chosen over rejected)
    reward_accuracy = (policy_diff > 0).float().mean()

    diagnostics = {
        "loss:sum": loss.item(),
        "reward_accuracy": reward_accuracy.item(),
        "mean_policy_chosen_score": policy_chosen_score.mean().item(),
        "mean_policy_rejected_score": policy_rejected_score.mean().item(),
        "mean_policy_diff": policy_diff.mean().item(),
        "mean_ref_diff": ref_diff.mean().item(),
    }

    return LossFnOutput(
        loss=loss,
        logprobs=policy_chosen_logprobs,  # Return chosen logprobs
        diagnostics=diagnostics
    )


def simple_contrastive_loss(
    model_outputs: torch.Tensor,
    loss_fn_inputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    temperature: float = 0.07,
) -> LossFnOutput:
    """
    Simple contrastive learning loss.

    This example shows a simpler custom loss that contrasts positive
    and negative examples.

    Args:
        model_outputs: Logits from model (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict containing:
            - positive_tokens: (batch_size, seq_len) - Positive example tokens
            - negative_tokens: (batch_size, seq_len) - Negative example tokens
        attention_mask: (batch_size, seq_len) - Attention mask
        temperature: Temperature for contrastive loss

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    positive_tokens = loss_fn_inputs.get("positive_tokens")
    negative_tokens = loss_fn_inputs.get("negative_tokens")

    if positive_tokens is None or negative_tokens is None:
        raise ValueError("contrastive requires 'positive_tokens' and 'negative_tokens'")

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Get log probs for positive examples
    positive_tokens_safe = positive_tokens.clone()
    positive_tokens_safe[positive_tokens == -100] = 0
    pos_logprobs = log_probs.gather(
        dim=-1,
        index=positive_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Get log probs for negative examples
    negative_tokens_safe = negative_tokens.clone()
    negative_tokens_safe[negative_tokens == -100] = 0
    neg_logprobs = log_probs.gather(
        dim=-1,
        index=negative_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Apply attention mask and compute mean scores
    pos_logprobs = pos_logprobs * attention_mask
    neg_logprobs = neg_logprobs * attention_mask

    num_tokens = attention_mask.sum()
    pos_score = pos_logprobs.sum() / num_tokens
    neg_score = neg_logprobs.sum() / num_tokens

    # Contrastive loss: maximize gap between positive and negative
    logits = torch.stack([pos_score, neg_score]) / temperature
    labels = torch.tensor([0], device=logits.device)  # Positive is index 0

    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), labels)

    diagnostics = {
        "loss:sum": loss.item(),
        "positive_score": pos_score.item(),
        "negative_score": neg_score.item(),
        "score_gap": (pos_score - neg_score).item(),
    }

    return LossFnOutput(
        loss=loss,
        logprobs=pos_logprobs,
        diagnostics=diagnostics
    )


async def main():
    print("=== Custom Loss Function Example ===\n")

    # Step 1: Register custom loss functions
    print("Step 1: Registering custom loss functions...")
    register_custom_loss("dpo", dpo_loss)
    register_custom_loss("contrastive", simple_contrastive_loss)
    print("✓ Registered: dpo, contrastive\n")

    # Step 2: Verify registration
    available_losses = loss_functions.LOSS_REGISTRY.list_available()
    print(f"Available loss functions: {available_losses}\n")

    # Step 3: Use custom loss in training
    print("Step 3: Training with custom DPO loss...\n")

    # Initialize backend
    hf_backend.initialize_base_model("HuggingFaceTB/SmolLM2-135M-Instruct")

    # Create LoRA adapter
    model_id = "dpo-model"
    lora_config = hf_backend.LoraConfigParams(r=16, lora_alpha=32)
    hf_backend.create_lora_adapter(model_id, lora_config)

    # Example DPO training data
    # In practice, you'd have pairs of (prompt, chosen_response, rejected_response)
    dpo_data = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}  # Chosen response
        ]
    ]

    # For DPO, you need reference model log probs and rejected completions
    # This is a simplified example - in practice you'd compute these from your data
    loss_fn_inputs = {
        "chosen_tokens": [[220, 17, 10, 17, 17159, 220, 19, 13]],  # Tokenized chosen response
        "rejected_tokens": [[220, 17, 10, 17, 284, 220, 20, 13]],  # Tokenized rejected response (wrong answer)
        "reference_chosen_logprobs": [[-0.5, -0.4, -0.3, -0.4, -0.5, -0.3, -0.4, -0.2]],
        "reference_rejected_logprobs": [[-0.5, -0.4, -0.3, -0.4, -0.5, -0.3, -0.4, -0.2]],
    }

    # Forward-backward with custom DPO loss
    for step in range(3):
        print(f"--- Training Step {step + 1} ---")

        result = hf_backend.forward_backward(
            model_id=model_id,
            data=dpo_data,
            loss_fn="dpo",  # Use our custom loss!
            loss_fn_inputs=loss_fn_inputs
        )

        print(f"Loss: {result['loss']:.4f}")
        print(f"Metrics: {result['metrics']}")

        # Optimizer step
        optim_result = hf_backend.optim_step(
            model_id=model_id,
            adam_params=hf_backend.AdamParams(learning_rate=5e-4)
        )
        print(f"✓ Optimizer step complete\n")

    print("\n=== Custom Loss Training Complete! ===\n")

    # Summary
    print("Key Takeaways:")
    print("1. Define loss function with signature:")
    print("   loss_fn(model_outputs, loss_fn_inputs, attention_mask) -> LossFnOutput")
    print("2. Register with: register_custom_loss(name, loss_fn)")
    print("3. Use in training by passing loss_fn=name to forward_backward()")
    print("4. Provide any required inputs via loss_fn_inputs parameter")


if __name__ == "__main__":
    asyncio.run(main())
