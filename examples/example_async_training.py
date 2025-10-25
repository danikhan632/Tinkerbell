#!/usr/bin/env python3
"""
Example: Async Training for Maximum Throughput

This example demonstrates using Tinkerbell's async API to maximize
training throughput by submitting multiple requests in parallel.

Based on Tinker async pattern:
https://tinker-docs.thinkingmachines.ai/async

Performance gain: ~3-5x faster than sync for batch training!
"""

import requests
import time
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def retrieve_future(future_id: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Wait for future to complete and return result.

    Args:
        future_id: The request_id from async endpoint
        timeout: Maximum seconds to wait

    Returns:
        Result dictionary when job completes

    Raises:
        TimeoutError: If job doesn't complete in time
        Exception: If job fails
    """
    start = time.time()

    while True:
        response = requests.post(f"{BASE_URL}/retrieve_future", json={
            "request_id": future_id
        })

        if response.status_code == 200:
            # Job completed successfully
            return response.json()

        elif response.status_code == 202:
            # Still pending
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"Future {future_id} timed out after {elapsed:.1f}s")

            time.sleep(0.5)  # Poll every 500ms

        else:
            # Error
            raise Exception(f"Future {future_id} failed: {response.text}")


def example_sync_training():
    """Example: Sync training (simple but slow)."""
    print("\n" + "="*70)
    print("Example 1: SYNC Training (Simple but Slow)")
    print("="*70 + "\n")

    # Create adapter
    print("Creating LoRA adapter...")
    response = requests.post(f"{BASE_URL}/api/v1/add_lora", json={
        "base_model": "base",
        "rank": 16,
        "alpha": 32
    })

    # Check response status
    if response.status_code != 200:
        print(f"✗ Failed to create adapter: {response.text}")
        return 0

    result = response.json()
    if "model_id" not in result:
        print(f"✗ Response missing model_id: {result}")
        return 0

    model_id = result["model_id"]
    print(f"✓ Created: {model_id}\n")

    # Training data
    training_data = [[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]]

    # Sync training - blocks on each call
    print("Training 3 steps (SYNC - sequential)...")
    start_time = time.time()

    for step in range(1, 4):
        print(f"\nStep {step}/3:")

        # Forward-backward (BLOCKS until complete)
        print("  Running forward-backward...")
        response = requests.post(f"{BASE_URL}/api/v1/forward_backward", json={
            "model_id": model_id,
            "data": training_data,
            "loss_fn": "cross_entropy"
        })
        result = response.json()
        print(f"  Loss: {result.get('loss', 0):.4f}")

        # Optimizer step (BLOCKS until complete)
        print("  Applying optimizer...")
        requests.post(f"{BASE_URL}/api/v1/optim_step", json={
            "model_id": model_id,
            "adam_params": {"learning_rate": 0.001}
        })
        print("  ✓ Step complete")

    elapsed = time.time() - start_time
    print(f"\n✓ Sync training complete in {elapsed:.2f}s")
    print(f"  Average per step: {elapsed/3:.2f}s")

    return elapsed


def example_async_training():
    """Example: Async training (complex but FAST!)."""
    print("\n" + "="*70)
    print("Example 2: ASYNC Training (Maximum Throughput)")
    print("="*70 + "\n")

    # Create adapter (async)
    print("Creating LoRA adapter (async)...")
    response = requests.post(f"{BASE_URL}/api/v1/add_lora_async", json={
        "base_model": "base",
        "rank": 16,
        "alpha": 32
    })
    future_id = response.json()["request_id"]
    result = retrieve_future(future_id)
    model_id = result["model_id"]
    print(f"✓ Created: {model_id}\n")

    # Training data
    training_data = [[
        {"role": "user", "content": "Hello async!"},
        {"role": "assistant", "content": "Hi there async!"}
    ]]

    # ASYNC training - submit all at once!
    print("Training 3 steps (ASYNC - parallel)...")
    start_time = time.time()

    # Step 1: Submit ALL forward-backward passes immediately
    print("\nSubmitting all forward-backward passes...")
    fwdbwd_futures = []

    for step in range(1, 4):
        response = requests.post(f"{BASE_URL}/api/v1/forward_backward_async", json={
            "model_id": model_id,
            "data": training_data,
            "loss_fn": "cross_entropy"
        })
        future_id = response.json()["request_id"]
        fwdbwd_futures.append(future_id)
        print(f"  ✓ Submitted step {step}")

    submit_time = time.time() - start_time
    print(f"\nAll steps submitted in {submit_time:.2f}s!")
    print("Now they're all processing in parallel...\n")

    # Step 2: Retrieve results as they complete
    print("Retrieving results:")
    for i, future_id in enumerate(fwdbwd_futures, 1):
        result = retrieve_future(future_id)
        print(f"  Step {i}: Loss={result.get('loss', 0):.4f} ✓")

        # Apply optimizer (async)
        response = requests.post(f"{BASE_URL}/api/v1/optim_step_async", json={
            "model_id": model_id,
            "adam_params": {"learning_rate": 0.001}
        })
        optim_future = response.json()["request_id"]
        retrieve_future(optim_future)

    elapsed = time.time() - start_time
    print(f"\n✓ Async training complete in {elapsed:.2f}s")
    print(f"  Average per step: {elapsed/3:.2f}s")

    return elapsed


def example_async_batched():
    """Example: Process batches asynchronously (MAXIMUM throughput!)."""
    print("\n" + "="*70)
    print("Example 3: Batched Async Training (MAXIMUM Throughput)")
    print("="*70 + "\n")

    # Create adapter
    print("Creating LoRA adapter...")
    response = requests.post(f"{BASE_URL}/api/v1/add_lora_async", json={
        "base_model": "base",
        "rank": 16,
        "alpha": 32
    })
    future_id = response.json()["request_id"]
    result = retrieve_future(future_id)
    model_id = result["model_id"]
    print(f"✓ Created: {model_id}\n")

    # Multiple batches of training data
    batches = [
        [[{"role": "user", "content": f"Batch {i}"}]]
        for i in range(5)
    ]

    print(f"Processing {len(batches)} batches asynchronously...")
    start_time = time.time()

    # Submit ALL batches at once!
    print("\nSubmitting all batches...")
    futures = []

    for i, batch in enumerate(batches):
        response = requests.post(f"{BASE_URL}/api/v1/forward_backward_async", json={
            "model_id": model_id,
            "data": batch,
            "loss_fn": "cross_entropy"
        })
        futures.append(response.json()["request_id"])
        print(f"  ✓ Submitted batch {i+1}")

    submit_time = time.time() - start_time
    print(f"\nAll {len(batches)} batches submitted in {submit_time:.2f}s!")
    print("Processing in parallel...\n")

    # Retrieve all results
    print("Retrieving results:")
    for i, future_id in enumerate(futures, 1):
        result = retrieve_future(future_id)
        loss = result.get('loss', 0)
        print(f"  Batch {i}: Loss={loss:.4f} ✓")

    elapsed = time.time() - start_time
    print(f"\n✓ Processed {len(batches)} batches in {elapsed:.2f}s")
    print(f"  Average per batch: {elapsed/len(batches):.2f}s")
    print(f"  Throughput: {len(batches)/elapsed:.2f} batches/sec")


def main():
    """Run all async examples."""
    print("\n" + "="*70)
    print("Tinkerbell Async API Examples")
    print("="*70)
    print("\nDemonstrating the power of async training!")
    print("Based on: https://tinker-docs.thinkingmachines.ai/async\n")

    # Check server
    try:
        response = requests.get(f"{BASE_URL}/healthz")
        print(f"✓ Server is healthy: {response.json()}\n")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nPlease start the server first:")
        print("  export TRAINING_BACKEND=megatron-bridge")
        print("  ./scripts/run_server_4gpu.sh")
        return

    # Run examples
    sync_time = example_sync_training()
    async_time = example_async_training()

    # Show speedup
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\nSync training:  {sync_time:.2f}s")
    print(f"Async training: {async_time:.2f}s")
    speedup = sync_time / async_time if async_time > 0 else 1
    print(f"\n🚀 Async is {speedup:.2f}x faster!")

    # Run batched example
    example_async_batched()

    # Summary
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. SYNC endpoints (e.g., /api/v1/forward_backward):
   - Simple to use
   - Block until complete
   - Good for learning/scripts

2. ASYNC endpoints (e.g., /api/v1/forward_backward_async):
   - Return future immediately
   - Non-blocking
   - MUCH faster for batches
   - Use /retrieve_future to get results

3. PERFORMANCE PATTERN:
   - Submit ALL requests first (async)
   - Then retrieve ALL results
   - This maximizes GPU utilization!

4. WHEN TO USE:
   - Sync: Simple scripts, learning
   - Async: Production, batches, max throughput
""")

    print("="*70)
    print("✓ Examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
