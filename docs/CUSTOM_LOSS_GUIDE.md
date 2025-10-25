# Custom Loss Functions Guide

Tinkerbell allows you to define custom loss functions on the client side and automatically upload them to the server using a simple decorator pattern.

## Quick Start

```python
from tinker_client import TinkerClient, custom_loss, LossFnOutput
import torch

# 1. Create a client
client = TinkerClient("http://localhost:8000")

# 2. Define your custom loss with the @custom_loss decorator
@custom_loss(client, name="my_custom_loss")
def my_loss(model_outputs, loss_fn_inputs, attention_mask):
    # Your custom loss logic here
    target_tokens = loss_fn_inputs["target_tokens"]

    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)
    target_logprobs = log_probs.gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)

    loss = (-target_logprobs * attention_mask).sum()

    return LossFnOutput(
        loss=loss,
        logprobs=target_logprobs,
        diagnostics={"loss": loss.item()}
    )

# 3. Use it in training!
result = hf_backend.forward_backward(
    model_id="my-model",
    data=training_data,
    loss_fn="my_custom_loss",
    loss_fn_inputs={"target_tokens": ...}
)
```

## How It Works

1. **Define**: Write your loss function as a normal Python function
2. **Decorate**: Add `@custom_loss(client, name="...")` above your function
3. **Upload**: The decorator automatically serializes and uploads your function to the server
4. **Use**: Reference the loss by name in your training calls

## Loss Function Signature

Your custom loss function must follow this signature:

```python
def my_loss(
    model_outputs: torch.Tensor,      # (batch_size, seq_len, vocab_size)
    loss_fn_inputs: Dict[str, torch.Tensor],  # Your custom inputs
    attention_mask: torch.Tensor,     # (batch_size, seq_len)
    **kwargs                          # Optional hyperparameters
) -> LossFnOutput:
    """
    Your loss function.

    Args:
        model_outputs: Logits from the model
        loss_fn_inputs: Dictionary containing your custom inputs (e.g., target_tokens, advantages, etc.)
        attention_mask: Mask indicating valid tokens vs padding
        **kwargs: Any additional hyperparameters

    Returns:
        LossFnOutput with three fields:
            - loss: Scalar tensor representing the loss value
            - logprobs: Log probabilities for tokens (can be any relevant logprobs)
            - diagnostics: Dict of metrics/diagnostics for logging
    """
    pass
```

### Input Parameters

- **model_outputs**: Logits from the forward pass, shape `(batch_size, seq_len, vocab_size)`
- **loss_fn_inputs**: A dictionary containing whatever data your loss needs. Common keys:
  - `target_tokens`: Token IDs for supervised learning
  - `weights`: Per-token weights for masking
  - `logprobs`: Reference log probs for RL
  - `advantages`: Advantage values for policy gradient methods
  - Any other custom data your loss requires
- **attention_mask**: Binary mask indicating which tokens are valid (1) vs padding (0)
- **kwargs**: Optional hyperparameters you can pass (e.g., `beta=0.1`, `temperature=0.07`)

### Return Value

Must return a `LossFnOutput` object with:
- **loss**: The scalar loss value (will be used for backward pass)
- **logprobs**: Log probabilities (typically for the target tokens)
- **diagnostics**: A dictionary of metrics that will be logged (e.g., accuracy, perplexity, etc.)

## Example: DPO (Direct Preference Optimization)

```python
@custom_loss(client, name="dpo")
def dpo_loss(model_outputs, loss_fn_inputs, attention_mask, beta=0.1):
    """DPO loss for preference learning."""

    # Extract inputs
    chosen_tokens = loss_fn_inputs["chosen_tokens"]
    rejected_tokens = loss_fn_inputs["rejected_tokens"]
    ref_chosen_logprobs = loss_fn_inputs["reference_chosen_logprobs"]
    ref_rejected_logprobs = loss_fn_inputs["reference_rejected_logprobs"]

    # Compute log probs
    log_probs = torch.nn.functional.log_softmax(model_outputs, dim=-1)

    # Get log probs for chosen responses
    policy_chosen_logprobs = log_probs.gather(
        dim=-1,
        index=chosen_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # Get log probs for rejected responses
    policy_rejected_logprobs = log_probs.gather(
        dim=-1,
        index=rejected_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # Apply attention mask
    policy_chosen_logprobs = policy_chosen_logprobs * attention_mask
    policy_rejected_logprobs = policy_rejected_logprobs * attention_mask

    # Compute per-example scores
    num_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    policy_chosen_score = policy_chosen_logprobs.sum(dim=1) / num_tokens.squeeze()
    policy_rejected_score = policy_rejected_logprobs.sum(dim=1) / num_tokens.squeeze()
    ref_chosen_score = ref_chosen_logprobs.sum(dim=1) / num_tokens.squeeze()
    ref_rejected_score = ref_rejected_logprobs.sum(dim=1) / num_tokens.squeeze()

    # DPO loss: -log sigmoid(beta * [policy_diff - ref_diff])
    policy_diff = policy_chosen_score - policy_rejected_score
    ref_diff = ref_chosen_score - ref_rejected_score
    logits = beta * (policy_diff - ref_diff)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # Compute metrics
    reward_accuracy = (policy_diff > 0).float().mean()

    return LossFnOutput(
        loss=loss,
        logprobs=policy_chosen_logprobs,
        diagnostics={
            "loss": loss.item(),
            "reward_accuracy": reward_accuracy.item(),
            "mean_policy_diff": policy_diff.mean().item(),
        }
    )

# Use in training
result = hf_backend.forward_backward(
    model_id="my-model",
    data=preference_data,
    loss_fn="dpo",
    loss_fn_inputs={
        "chosen_tokens": chosen_tok,
        "rejected_tokens": rejected_tok,
        "reference_chosen_logprobs": ref_chosen,
        "reference_rejected_logprobs": ref_rejected,
    }
)
```

## Built-in Loss Functions

Tinkerbell comes with three built-in loss functions:

### 1. cross_entropy
Standard supervised learning loss (negative log-likelihood).

```python
result = hf_backend.forward_backward(
    model_id="my-model",
    data=training_data,
    loss_fn="cross_entropy"
)
```

**Inputs**: Automatically uses `target_tokens` from the data.

### 2. importance_sampling
REINFORCE with importance sampling for off-policy RL.

```python
result = hf_backend.forward_backward(
    model_id="my-model",
    data=rl_data,
    loss_fn="importance_sampling",
    loss_fn_inputs={
        "target_tokens": sampled_tokens,
        "logprobs": reference_logprobs,
        "advantages": advantage_values
    }
)
```

### 3. ppo
Proximal Policy Optimization with clipping (epsilon=0.2).

```python
result = hf_backend.forward_backward(
    model_id="my-model",
    data=rl_data,
    loss_fn="ppo",
    loss_fn_inputs={
        "target_tokens": sampled_tokens,
        "logprobs": reference_logprobs,
        "advantages": advantage_values
    }
)
```

## Listing Available Loss Functions

```python
from tinker_client import TinkerClient

client = TinkerClient("http://localhost:8000")
losses = client.list_loss_functions()

print("Available loss functions:")
for name in losses["available_loss_functions"]:
    print(f"  - {name}")
```

## Advanced Features

### Hyperparameters

You can add hyperparameters to your loss function:

```python
@custom_loss(client, name="my_loss")
def my_loss(model_outputs, loss_fn_inputs, attention_mask,
            temperature=0.07, weight_decay=0.01):
    # Use temperature and weight_decay in your loss computation
    ...
```

### Multiple Custom Losses

You can register multiple custom losses:

```python
@custom_loss(client, name="loss_a")
def loss_a(model_outputs, loss_fn_inputs, attention_mask):
    ...

@custom_loss(client, name="loss_b")
def loss_b(model_outputs, loss_fn_inputs, attention_mask):
    ...

# Use either one in training
result = hf_backend.forward_backward(..., loss_fn="loss_a")
# or
result = hf_backend.forward_backward(..., loss_fn="loss_b")
```

### Complex Diagnostics

Return rich diagnostics for monitoring:

```python
return LossFnOutput(
    loss=loss,
    logprobs=logprobs,
    diagnostics={
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "perplexity": perplexity.item(),
        "mean_logprob": mean_logprob.item(),
        "std_logprob": std_logprob.item(),
        "grad_norm": grad_norm.item(),
        # Any other metrics you want to track
    }
)
```

## Requirements

Custom loss functions require the `cloudpickle` library for serialization:

```bash
pip install cloudpickle
```

## Examples

See the `examples/` directory for complete working examples:

- **example_simple_custom_loss.py**: Minimal example showing basic usage
- **example_custom_loss_decorator.py**: Comprehensive examples including DPO, contrastive learning, and KL-regularized losses

## Troubleshooting

### Import Error: cloudpickle not found
```
pip install cloudpickle
```

### Server Connection Error
Make sure the Tinkerbell server is running:
```bash
python src/app.py
```

### Loss Function Not Found
Check that your loss was registered successfully:
```python
losses = client.list_loss_functions()
print(losses["available_loss_functions"])
```

### Serialization Error
Make sure all dependencies used in your loss function are available on the server. Avoid using local variables from outside the function scope.

## Architecture

The custom loss system works as follows:

1. **Client Side**: The `@custom_loss` decorator serializes your function using `cloudpickle` and base64 encoding
2. **Upload**: The serialized function is sent to the server via the `/register_custom_loss` endpoint
3. **Server Side**: The worker deserializes the function and registers it with the loss function registry
4. **Training**: When you call `forward_backward(loss_fn="my_loss")`, the server looks up and executes your custom loss

This architecture allows you to write loss functions in pure Python with full PyTorch support, while the server handles all the distributed training complexity.
