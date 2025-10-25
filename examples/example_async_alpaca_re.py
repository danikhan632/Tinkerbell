#!/usr/bin/env python3
"""
Example: Async RL Training with vLLM Sampling + Reward-based Training

This example demonstrates a simplified RL training loop:
1. Sample completions from the model using vLLM (fast inference)
2. Compute rewards based on some criteria (e.g., length, sentiment, etc.)
3. Train the model using those rewards with async forward-backward

This is inspired by the RLHF pattern from Megatron-Bridge's rlhf_with_bridge.py
but simplified for the Tinkerbell async API.

Flow:
- vLLM generates completions (policy rollout)
- Simple reward function scores the completions
- Async training updates the policy based on rewards
"""

import requests
import time
import random
from typing import Dict, Any, List


BASE_URL = "http://localhost:8000"

# Instruction-following prompts from Alpaca dataset
ALPACA_INSTRUCTIONS = [
    "Give three tips for staying healthy.",
    "What is the capital of France?",
    "Generate a poem with 10 lines.",
    "Give an example of a metaphor that uses the following object: Stars",
    "Describe the sound of the given object: Wind chime",
    "Explain why the following statement is true: A successful sales pitch should engage the potential buyer.",
    "Compare and contrast winter and summer.",
    "Rewrite the following sentence using active voice: The news report was read by the captain.",
    "Classify the following incident as a breach of protocol. Output 1 for breach, and 0 for no breach: Using a school laptop for personal use",
    "Rewrite the sentence to provide more clarity and flow: Making the decision to rent a house was a wise choice"
]


def retrieve_future(future_id: str, timeout: int = 120) -> Dict[str, Any]:
    """Wait for async future to complete."""
    start = time.time()

    while True:
        response = requests.post(f"{BASE_URL}/retrieve_future", json={
            "request_id": future_id
        })

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"Future {future_id} timed out after {elapsed:.1f}s")
            time.sleep(0.5)
        else:
            raise Exception(f"Future {future_id} failed: {response.text}")


def sample_with_vllm(model_id: str, prompts: List[str], max_tokens: int = 100) -> List[Dict[str, Any]]:
    """
    Generate completions using vLLM (fast inference engine).

    This is the "policy rollout" phase in RL training.
    """
    print(f"\n🎲 Sampling completions with vLLM (model: {model_id})...")

    results = []
    for i, prompt in enumerate(prompts, 1):
        # Use async sampling endpoint
        # vLLM expects 'prompts' (list of strings) not 'messages'
        response = requests.post(f"{BASE_URL}/api/v1/asample", json={
            "model_id": model_id,  # Use model_id for LoRA adapter
            "prompts": [prompt],    # vLLM expects list of prompt strings
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        })

        if response.status_code != 200:
            print(f"  Sample {i}: Failed - {response.text}")
            continue

        future_id = response.json()["request_id"]

        try:
            result = retrieve_future(future_id, timeout=30)
            completion = result.get("completions", [""])[0] if "completions" in result else ""

            results.append({
                "prompt": prompt,
                "completion": completion,
                "full_text": prompt + "\n\n" + completion
            })

            print(f"  Sample {i}: ✓ Generated {len(completion)} chars")

        except Exception as e:
            print(f"  Sample {i}: Failed - {e}")

    return results


def compute_rewards(samples: List[Dict[str, Any]]) -> List[float]:
    """
    Compute rewards for generated completions.

    This is a simple reward function for demonstration.
    In production RLHF, you'd use:
    - Trained reward model (e.g., sentiment classifier)
    - Human preferences (Bradley-Terry model)
    - Rule-based heuristics (safety, helpfulness)
    """
    print(f"\n💰 Computing rewards for {len(samples)} samples...")

    rewards = []
    for i, sample in enumerate(samples, 1):
        completion = sample["completion"]

        # Simple reward heuristics (replace with real reward model)
        reward = 0.0

        # Reward 1: Length (prefer reasonable length responses)
        length = len(completion)
        if 50 <= length <= 500:
            reward += 0.3
        elif length > 500:
            reward += 0.1  # Penalize overly long
        else:
            reward += 0.0  # Penalize too short

        # Reward 2: Contains newlines (structured response)
        if "\n" in completion:
            reward += 0.2

        # Reward 3: Not empty
        if len(completion.strip()) > 10:
            reward += 0.3

        # Reward 4: Random noise (simulate reward model uncertainty)
        reward += random.uniform(-0.1, 0.1)

        # Normalize to [0, 1]
        reward = max(0.0, min(1.0, reward))

        rewards.append(reward)
        print(f"  Sample {i}: reward={reward:.3f} (length={length})")

    return rewards


def async_rl_training(model_id: str, samples: List[Dict[str, Any]], rewards: List[float], kl_coeff: float = 0.1):
    """
    Train the model using async forward-backward with rewards and KL divergence penalty.

    This uses importance sampling / REINFORCE-style loss with KL divergence constraint:
    - Higher rewards → lower loss (reinforce good behaviors)
    - KL penalty prevents policy from deviating too much from base model
    - Training data is the generated completions

    Args:
        model_id: LoRA adapter ID
        samples: List of samples with prompt and completion
        rewards: List of rewards for each sample
        kl_coeff: KL divergence penalty coefficient (default: 0.1)
    """
    print(f"\n🔧 Async RL training on {len(samples)} samples (KL coeff: {kl_coeff})...")
    print("Submitting training jobs...")

    start_time = time.time()
    training_futures = []

    for i, (sample, reward) in enumerate(zip(samples, rewards), 1):
        # Format as training data (prompt + completion)
        training_messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]}
        ]

        # Submit async forward-backward with reward-based loss + KL penalty
        # The loss_fn_inputs passes the reward and KL parameters
        response = requests.post(f"{BASE_URL}/api/v1/forward_backward_async", json={
            "model_id": model_id,
            "data": [training_messages],
            "loss_fn": "importance_sampling",  # RL-style loss
            "loss_fn_inputs": {
                "rewards": [reward],
                "prompts": [sample["prompt"]],      # For KL computation
                "completions": [sample["completion"]],  # For KL computation
                "kl_coeff": kl_coeff,  # KL penalty coefficient
                # In real RLHF, you'd also pass:
                # "old_log_probs": [...],  # from policy before update
                # "advantages": [...],     # GAE or similar
            }
        })

        if response.status_code == 200:
            future_id = response.json()["request_id"]
            training_futures.append((i, future_id, reward))
            print(f"  ✓ Submitted sample {i} (reward={reward:.3f})")
        else:
            print(f"  ✗ Failed sample {i}: {response.text}")

    submit_time = time.time() - start_time
    print(f"\n✓ All {len(training_futures)} training jobs submitted in {submit_time:.2f}s")
    print("Processing in parallel...\n")

    # Retrieve training results
    print("Training results:")
    total_loss = 0.0
    total_kl = 0.0
    successful = 0

    for i, future_id, reward in training_futures:
        try:
            result = retrieve_future(future_id, timeout=30)
            loss = result.get("loss", 0.0)
            base_loss = result.get("base_loss", loss)
            kl_penalty = result.get("kl_penalty", 0.0)
            kl_div = result.get("metrics", {}).get("kl_div", 0.0)

            total_loss += loss
            total_kl += kl_div
            successful += 1

            print(f"  Sample {i}: loss={loss:.4f} (base={base_loss:.4f}, kl_penalty={kl_penalty:.4f}), reward={reward:.3f}, kl={kl_div:.4f} ✓")

            # Apply optimizer step
            opt_response = requests.post(f"{BASE_URL}/api/v1/optim_step_async", json={
                "model_id": model_id,
                "adam_params": {"learning_rate": 0.0001}
            })

            if opt_response.status_code == 200:
                opt_future = opt_response.json()["request_id"]
                retrieve_future(opt_future, timeout=10)

        except Exception as e:
            print(f"  Sample {i}: Failed - {e}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / successful if successful > 0 else 0.0
    avg_kl = total_kl / successful if successful > 0 else 0.0

    print(f"\n✓ RL training complete in {elapsed:.2f}s")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Average KL div: {avg_kl:.4f}")
    print(f"  Success rate: {successful}/{len(training_futures)}")

    return avg_loss, avg_kl, elapsed


def main():
    """Run RL training loop: vLLM sampling → reward computation → async training."""
    print("\n" + "="*70)
    print("Tinkerbell: Async RL Training with vLLM Policy Sampling")
    print("="*70)
    print("""
This example demonstrates a simplified RL training loop:

1. 🎲 ROLLOUT: Sample completions from model using vLLM (fast!)
2. 💰 REWARD:  Compute rewards for each completion
3. 🔧 TRAIN:   Update policy using async forward-backward with rewards
4. 🔁 REPEAT:  Multiple training iterations

Pattern inspired by: Megatron-Bridge rlhf_with_bridge.py
""")

    # Check server
    try:
        response = requests.get(f"{BASE_URL}/healthz")
        print(f"✓ Server is healthy: {response.json()}\n")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nPlease start server with vLLM enabled:")
        print("  export TRAINING_BACKEND=megatron-bridge")
        print("  ./scripts/run_server_4gpu.sh")
        return

    # Create LoRA adapter for training
    print("Creating LoRA adapter...")
    response = requests.post(f"{BASE_URL}/api/v1/add_lora_async", json={
        "base_model": "base",
        "rank": 16,
        "alpha": 32
    })

    if response.status_code != 200:
        print(f"✗ Failed to create adapter: {response.text}")
        return

    future_id = response.json()["request_id"]
    result = retrieve_future(future_id)
    model_id = result["model_id"]
    print(f"✓ Created adapter: {model_id}\n")

    # Training configuration
    num_iterations = 5  # Number of RL training loops
    samples_per_iteration = 5  # Samples per loop
    selected_prompts = ALPACA_INSTRUCTIONS[:samples_per_iteration]

    # Track metrics across iterations
    all_rewards = []
    all_losses = []
    total_samples = 0
    total_time = 0

    print("="*70)
    print(f"TRAINING CONFIGURATION")
    print("="*70)
    print(f"  RL Iterations: {num_iterations}")
    print(f"  Samples per iteration: {samples_per_iteration}")
    print(f"  Total samples: {num_iterations * samples_per_iteration}")
    print(f"  Prompts: {len(selected_prompts)}")
    print("="*70 + "\n")

    # Multi-iteration RL training loop
    for iteration in range(1, num_iterations + 1):
        print("\n" + "="*70)
        print(f"ITERATION {iteration}/{num_iterations}")
        print("="*70)

        # Phase 1: Sample from policy (vLLM)
        samples = sample_with_vllm(model_id, selected_prompts, max_tokens=150)

        if len(samples) == 0:
            print(f"\n✗ Iteration {iteration}: No samples generated. Skipping...")
            continue

        # Phase 2: Compute rewards
        rewards = compute_rewards(samples)

        # Phase 3: Train with async API (with KL divergence constraint)
        kl_coeff = 0.1  # KL penalty coefficient
        avg_loss, avg_kl, train_time = async_rl_training(model_id, samples, rewards, kl_coeff=kl_coeff)

        # Track metrics
        all_rewards.extend(rewards)
        all_losses.append(avg_loss)
        total_samples += len(samples)
        total_time += train_time

        # Show iteration summary
        avg_reward_this_iter = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"\n📊 Iteration {iteration} Summary:")
        print(f"  Avg Reward: {avg_reward_this_iter:.3f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg KL Div: {avg_kl:.4f}")
        print(f"  Time: {train_time:.2f}s")

    # Final average across all iterations
    avg_reward_overall = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    avg_loss_overall = sum(all_losses) / len(all_losses) if all_losses else 0.0

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"""
✓ RL Training Complete!

Model: {model_id}
Total Iterations: {num_iterations}
Total Samples Trained: {total_samples}
Total Training Time: {total_time:.2f}s

PERFORMANCE METRICS:
  Overall Avg Reward: {avg_reward_overall:.3f}
  Overall Avg Loss: {avg_loss_overall:.4f}
  Throughput: {total_samples/total_time:.2f} samples/sec
  Time per iteration: {total_time/num_iterations:.2f}s

""")

    print("="*70)
    print("✓ Example complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
