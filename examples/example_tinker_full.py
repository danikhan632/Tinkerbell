"""
Complete example using Tinker SDK with the Flask server.

This demonstrates:
1. Creating a ServiceClient
2. Training with different loss functions
3. Saving and loading checkpoints
4. Using LoRA adapters

Requirements:
    pip install tinker-sdk

Usage:
    # Start the Flask server first
    python app.py

    # Then run this example
    python example_tinker_full.py
"""

import asyncio
import tinker


async def main():
    print("=== Tinker SDK Full Example ===\n")

    # Create service client
    service_client = tinker.ServiceClient(base_url="http://localhost:8000")

    # Check server health
    health_response = await service_client.healthz()
    print(f"✓ Server is healthy: {health_response.status}")

    # Get server capabilities
    capabilities = await service_client.get_server_capabilities()
    print(f"✓ Server capabilities loaded")
    print(f"  Supported models: {len(capabilities.supported_models)}\n")

    # ============================================================================
    # Example 1: Create Training Client for Supervised Learning
    # ============================================================================
    print("=== Creating Training Client for Supervised Learning ===\n")

    training_client = await service_client.create_training_client(
        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_id="math-tutor",
        # LoRA config (optional)
        lora_config={
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
    )

    print(f"✓ Created training client for model: math-tutor\n")

    # Training data
    math_data = [
        tinker.Datum(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        ),
        tinker.Datum(
            messages=[
                {"role": "user", "content": "What is 5+3?"},
                {"role": "assistant", "content": "5+3 equals 8."}
            ]
        ),
        tinker.Datum(
            messages=[
                {"role": "user", "content": "What is 10-7?"},
                {"role": "assistant", "content": "10-7 equals 3."}
            ]
        )
    ]

    # Training loop with cross-entropy loss
    print("Training with cross_entropy loss...\n")

    for step in range(5):
        print(f"Step {step + 1}/5")

        # Forward-backward pass
        fwdbwd_result = await training_client.forward_backward(
            data=math_data,
            loss_fn="cross_entropy"
        )

        # Extract loss
        loss_value = fwdbwd_result.loss_fn_outputs.get('loss', 0)
        print(f"  Loss: {loss_value:.4f}")

        # Optimizer step
        await training_client.optim_step(
            adam_params=tinker.AdamParams(learning_rate=5e-4)
        )

        print(f"  ✓ Optimizer step complete\n")

    print("✓ Supervised learning complete!\n")

    # ============================================================================
    # Example 2: RL Training with PPO
    # ============================================================================
    print("=== Creating Training Client for RL (PPO) ===\n")

    rl_training_client = await service_client.create_training_client(
        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_id="rl-agent",
        lora_config={"rank": 16, "alpha": 32}
    )

    print(f"✓ Created RL training client\n")

    # RL training data (simplified)
    rl_data = [
        tinker.Datum(
            messages=[
                {"role": "user", "content": "Solve: 10-3"},
                {"role": "assistant", "content": "10-3=7"}
            ]
        )
    ]

    # In a real RL setup, you would:
    # 1. Sample completions from the policy
    # 2. Compute rewards
    # 3. Calculate advantages
    # Here we use simulated values

    print("Training with PPO loss...\n")

    for step in range(3):
        print(f"Step {step + 1}/3")

        # Forward-backward with PPO
        # Note: In production, target_tokens, logprobs, and advantages
        # come from your RL pipeline
        fwdbwd_result = await rl_training_client.forward_backward(
            data=rl_data,
            loss_fn="ppo",
            loss_fn_inputs={
                "target_tokens": [[220, 16, 15, 489, 220, 18, 284, 220, 22]],
                "logprobs": [[-0.4, -0.5, -0.3, -0.4, -0.3, -0.5, -0.2, -0.4, -0.3]],
                "advantages": [[0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0]]
            }
        )

        loss_value = fwdbwd_result.loss_fn_outputs.get('loss', 0)
        print(f"  Loss: {loss_value:.4f}")

        # Extract PPO-specific metrics
        if 'metrics' in fwdbwd_result.loss_fn_outputs:
            metrics = fwdbwd_result.loss_fn_outputs['metrics']
            print(f"  Clip fraction: {metrics.get('clip_fraction', 0):.3f}")

        # Optimizer step
        await rl_training_client.optim_step(
            adam_params=tinker.AdamParams(learning_rate=5e-4)
        )

        print(f"  ✓ Optimizer step complete\n")

    print("✓ RL training complete!\n")

    # ============================================================================
    # Example 3: Save Models for Deployment
    # ============================================================================
    print("=== Saving Models for Deployment ===\n")

    # Save the supervised model
    try:
        supervised_sampler = await training_client.save_weights_and_get_sampling_client(
            name="math-tutor-v1"
        )
        print(f"✓ Saved math-tutor model for deployment")
    except Exception as e:
        print(f"Note: save_weights_and_get_sampling_client not yet implemented: {e}")

    # Save the RL model
    try:
        rl_sampler = await rl_training_client.save_weights_and_get_sampling_client(
            name="rl-agent-v1"
        )
        print(f"✓ Saved rl-agent model for deployment")
    except Exception as e:
        print(f"Note: save_weights_and_get_sampling_client not yet implemented: {e}")

    print()

    # ============================================================================
    # Example 4: Inference with Trained Models
    # ============================================================================
    print("=== Testing Inference ===\n")

    # Create sampling client
    try:
        sampling_client = await service_client.create_sampling_client(
            model_id="math-tutor"
        )

        # Generate completion
        result = await sampling_client.sample(
            prompt="What is 7+5?",
            max_tokens=50,
            temperature=0.7
        )

        print(f"Prompt: What is 7+5?")
        print(f"Response: {result.generated_text}")
    except Exception as e:
        print(f"Note: Inference example skipped: {e}")

    print()

    # ============================================================================
    # Summary
    # ============================================================================
    print("=== Training Summary ===\n")
    print("✓ Trained 2 models:")
    print("  1. math-tutor - Supervised learning with cross_entropy")
    print("  2. rl-agent - RL training with PPO")
    print()
    print("Next steps:")
    print("  - Save checkpoints regularly during training")
    print("  - Use save_weights_for_sampler() for deployment")
    print("  - Monitor metrics (loss, perplexity, clip_fraction)")
    print("  - Adjust learning rates based on performance")


if __name__ == "__main__":
    asyncio.run(main())
