"""
Example using Tinker client library to train with loss functions.

This demonstrates how to use the official Tinker SDK with the Flask server.
"""

import asyncio
from tinker import ServiceClient
from tinker.types import (
    TrainingForwardBackwardParams,
    TrainingOptimStepParams,
    AdamParams,
)


async def main():
    # Connect to the Flask server
    client = ServiceClient(base_url="http://localhost:8000")

    print("=== Tinker Client Example ===\n")

    # Check server health
    health = await client.healthz()
    print(f"Server status: {health.status}")

    # Get server capabilities
    capabilities = await client.get_server_capabilities()
    print(f"Supported models: {[m.model_id for m in capabilities.supported_models]}\n")

    # Training data
    training_data = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ],
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    ]

    print("=== Example 1: Supervised Learning with cross_entropy ===\n")

    model_id = "supervised-model"

    # Training loop
    for step in range(3):
        print(f"--- Step {step + 1} ---")

        # Forward-backward pass
        fwdbwd_params = TrainingForwardBackwardParams(
            model_id=model_id,
            data=training_data,
            loss_fn="cross_entropy"
        )

        fwdbwd_future = await client.fwdbwd(fwdbwd_params)
        fwdbwd_result = await fwdbwd_future.result()

        print(f"Loss: {fwdbwd_result.loss_fn_outputs}")
        if hasattr(fwdbwd_result, 'metrics'):
            print(f"Metrics: {fwdbwd_result.metrics}")

        # Optimizer step
        optim_params = TrainingOptimStepParams(
            model_id=model_id,
            adam_params=AdamParams(learning_rate=5e-4)
        )

        optim_future = await client.optim_step(optim_params)
        optim_result = await optim_future.result()

        print(f"Optimizer step complete\n")

    print("\n=== Example 2: RL Training with PPO ===\n")

    rl_model_id = "rl-model-ppo"

    # RL training data
    rl_data = [
        [
            {"role": "user", "content": "What is 10 - 3?"},
            {"role": "assistant", "content": "10 - 3 = 7"}
        ]
    ]

    # Simulated RL inputs (in practice, these come from your RL pipeline)
    target_tokens = [[220, 16, 15, 489, 220, 18, 284, 220, 22]]
    sampling_logprobs = [[-0.4, -0.5, -0.3, -0.4, -0.3, -0.5, -0.2, -0.4, -0.3]]
    advantages = [[0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0]]

    for step in range(3):
        print(f"--- Step {step + 1} ---")

        # Forward-backward with PPO loss
        fwdbwd_params = TrainingForwardBackwardParams(
            model_id=rl_model_id,
            data=rl_data,
            loss_fn="ppo",
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "logprobs": sampling_logprobs,
                "advantages": advantages
            }
        )

        fwdbwd_future = await client.fwdbwd(fwdbwd_params)
        fwdbwd_result = await fwdbwd_future.result()

        print(f"Loss: {fwdbwd_result.loss_fn_outputs}")
        if hasattr(fwdbwd_result, 'metrics'):
            metrics = fwdbwd_result.metrics
            print(f"Metrics:")
            print(f"  - Mean advantage: {metrics.get('mean_advantage', 'N/A')}")
            print(f"  - Mean ratio: {metrics.get('mean_ratio', 'N/A')}")
            print(f"  - Clip fraction: {metrics.get('clip_fraction', 'N/A')}")

        # Optimizer step
        optim_params = TrainingOptimStepParams(
            model_id=rl_model_id,
            adam_params=AdamParams(learning_rate=5e-4)
        )

        optim_future = await client.optim_step(optim_params)
        optim_result = await optim_future.result()

        print(f"Optimizer step complete\n")

    print("\n=== Example 3: Text Generation ===\n")

    # Generate with the supervised model
    from tinker.types import SamplingSampleParams, SamplingParams, Prompt, Chunk

    # Tokenize prompt (simplified - in practice use proper tokenizer)
    prompt_text = "What is Python?"
    # For this example, we'll use a simplified approach
    # In production, you'd tokenize properly

    sample_params = SamplingSampleParams(
        base_model="supervised-model",
        prompt=Prompt(chunks=[Chunk(tokens=[1, 2, 3])]),  # Placeholder tokens
        sampling_params=SamplingParams(
            max_tokens=50,
            temperature=0.7
        )
    )

    try:
        sample_future = await client.sample(sample_params)
        sample_result = await sample_future.result()
        print(f"Generated: {sample_result}")
    except Exception as e:
        print(f"Generation example skipped (requires proper tokenization): {e}")

    print("\n=== Training Complete! ===")


if __name__ == "__main__":
    asyncio.run(main())
