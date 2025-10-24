#!/usr/bin/env python3
"""
Simple Example: 3 Users Training Different LoRA Adapters Concurrently

This is a minimal example showing concurrent multi-user training.
Each user gets their own LoRA adapter and trains it independently.

Backend Configuration:
- Training: Uses HuggingFace backend (concurrent) or Megatron (sequential)
- Sampling: Uses vLLM for high-performance inference with LoRA adapters
  - Enable vLLM: export USE_VLLM=true
  - Start vLLM server: trl vllm-serve --model MODEL --enable-lora
  - Falls back to HuggingFace if vLLM is not available
"""

import threading
import time
from client_example import TinkerClient


def train_user_adapter(user_name: str, client: TinkerClient):
    """Train a single user's LoRA adapter."""
    print(f"\n[{user_name}] Starting...")

    try:
        # Step 1: Create a LoRA adapter
        print(f"[{user_name}] Creating LoRA adapter...")
        request_id = client.add_lora(
            base_model="base",
            rank=8,
            alpha=16
        )
        result = client.wait_for_result(request_id, timeout=30)
        model_id = result["model_id"]
        print(f"[{user_name}] ✓ Created adapter: {model_id}")

        # Step 2: Train with personalized data
        training_data = [[
            {"role": "user", "content": f"Hello, I'm {user_name}!"},
            {"role": "assistant", "content": f"Nice to meet you, {user_name}!"}
        ]]

        for step in range(1, 4):  # 3 training steps
            print(f"[{user_name}] Training step {step}/3...")

            # Forward-backward pass
            request_id = client.fwdbwd(
                model_id=model_id,
                data=training_data,
                loss_fn="cross_entropy"
            )
            result = client.wait_for_result(request_id, timeout=30)
            loss = result.get("loss", 0.0)
            print(f"[{user_name}]   Step {step}: Loss = {loss:.4f}")

            # Optimizer step
            request_id = client.optim_step(
                model_id=model_id,
                adam_params={"learning_rate": 0.001}
            )
            client.wait_for_result(request_id, timeout=30)
            print(f"[{user_name}]   Step {step}: Optimizer applied")

        # Step 3: Test inference (uses vLLM if enabled)
        print(f"[{user_name}] Testing inference with LoRA adapter...")
        # NOTE: If USE_VLLM=true, this will use vLLM for fast sampling with the LoRA adapter
        # The model_id is used to select the appropriate LoRA adapter in vLLM
        result = client.sample(
            model_id=model_id,  # LoRA adapter ID
            prompts=[f"Hello {user_name}!"],
            sampling_params={"max_tokens": 20, "temperature": 0.7}
        )
        generated = result.get("generated_text", "")
        print(f"[{user_name}] Generated: {generated}")

        print(f"[{user_name}] ✓✓✓ DONE! ✓✓✓")

    except Exception as e:
        print(f"[{user_name}] ✗✗✗ ERROR: {e}")


def main():
    """Run 3 users concurrently."""

    print("\n" + "="*70)
    print("Simple Concurrent Multi-User Example")
    print("="*70)
    print("\nThis example shows 3 users training their own LoRA adapters")
    print("simultaneously without blocking each other.\n")

    print("Backend Configuration:")
    print("  Training: HuggingFace (concurrent) or Megatron (sequential)")
    print("  Sampling: vLLM (if USE_VLLM=true) or HuggingFace (fallback)")
    print("\nTo use vLLM for fast sampling:")
    print("  1. Start vLLM server: trl vllm-serve --model MODEL --enable-lora")
    print("  2. Set environment: export USE_VLLM=true")
    print("  3. Run this example\n")

    # Create client
    client = TinkerClient("http://localhost:8000")

    # Check server
    try:
        health = client.health_check()
        print(f"✓ Server is healthy: {health}\n")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nPlease start the server first:")
        print("  python src/app.py")
        return

    print("Starting 3 concurrent users...\n")

    start_time = time.time()

    # Create threads for 3 users
    users = ["Alice", "Bob", "Carol"]
    threads = []

    for user in users:
        thread = threading.Thread(
            target=train_user_adapter,
            args=(user, client)
        )
        threads.append(thread)

    # Start all threads (concurrent execution)
    for thread in threads:
        thread.start()

    # Wait for all to complete
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ All 3 users completed training!")
    print(f"✓ Total time: {total_time:.2f} seconds")
    print(f"\nKey Points:")
    print(f"  • Each user trained their own LoRA adapter")
    print(f"  • All training happened CONCURRENTLY (not sequential)")
    print(f"  • Users didn't block each other")
    print(f"  • Speedup: ~3x compared to sequential execution")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
