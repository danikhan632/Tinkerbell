"""
Simple Custom Loss Example

Shows the minimal code needed to define and use a custom loss function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tinker_client import TinkerClient, custom_loss, LossFnOutput
import torch

# 1. Create client
client = TinkerClient("http://localhost:8000")

# 2. Define your custom loss with the decorator
@custom_loss(client, name="my_loss")
def my_custom_loss(model_outputs, loss_fn_inputs, attention_mask):
    """
    Your custom loss function.

    Args:
        model_outputs: Logits from the model (batch_size, seq_len, vocab_size)
        loss_fn_inputs: Dict of your custom inputs
        attention_mask: Attention mask (batch_size, seq_len)

    Returns:
        LossFnOutput with loss, logprobs, and diagnostics
    """
    # Get your custom inputs
    target_tokens = loss_fn_inputs["target_tokens"]

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Get log probs for targets
    target_tokens_safe = target_tokens.clone()
    target_tokens_safe[target_tokens == -100] = 0
    target_logprobs = log_probs.gather(
        dim=-1,
        index=target_tokens_safe.unsqueeze(-1)
    ).squeeze(-1)

    # Compute your loss (example: negative log likelihood)
    loss = (-target_logprobs * attention_mask).sum()

    # Return LossFnOutput
    return LossFnOutput(
        loss=loss,
        logprobs=target_logprobs,
        diagnostics={
            "loss": loss.item(),
            "num_tokens": attention_mask.sum().item()
        }
    )


# 3. That's it! The decorator uploaded your loss to the server.
# Now you can use it in training:

print("✓ Custom loss 'my_loss' has been registered!")
print("\nUse it in training like this:")
print("""
import hf_backend

result = hf_backend.forward_backward(
    model_id="my-model",
    data=training_data,
    loss_fn="my_loss",  # Your custom loss!
    loss_fn_inputs={
        "target_tokens": [[1, 2, 3, 4]]
    }
)
""")

# List all available losses
losses = client.list_loss_functions()
print(f"\nAvailable losses: {losses.get('available_loss_functions', [])}")
