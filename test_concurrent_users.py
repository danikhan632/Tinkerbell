#!/usr/bin/env python3
"""
Test script for concurrent multi-user LoRA training.

This script simulates multiple users training different LoRA adapters
concurrently to verify thread safety and concurrent processing.
"""

import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List


BASE_URL = "http://localhost:8000"


def create_lora_adapter(user_id: str, rank: int = 8) -> str:
    """Create a new LoRA adapter for a user."""
    print(f"[User {user_id}] Creating LoRA adapter...")

    response = requests.post(
        f"{BASE_URL}/add_lora",
        json={
            "base_model": "base",
            "rank": rank,
            "alpha": rank * 2
        }
    )

    future_id = response.json()["request_id"]

    # Wait for adapter creation
    while True:
        result = requests.post(
            f"{BASE_URL}/retrieve_future",
            json={"request_id": future_id}
        )

        if result.status_code == 200:
            model_id = result.json()["model_id"]
            print(f"[User {user_id}] LoRA adapter created: {model_id}")
            return model_id
        elif result.status_code == 202:
            time.sleep(0.1)
        else:
            raise Exception(f"Failed to create adapter: {result.text}")


def train_adapter(user_id: str, model_id: str, num_steps: int = 3) -> Dict:
    """Train a LoRA adapter with multiple forward-backward and optim steps."""
    print(f"[User {user_id}] Starting training for {num_steps} steps...")

    training_data = [
        [
            {"role": "user", "content": f"Hello from user {user_id}!"},
            {"role": "assistant", "content": f"Hi user {user_id}, I'm your personalized assistant."}
        ]
    ]

    metrics = []

    for step in range(num_steps):
        # Forward-backward pass
        print(f"[User {user_id}] Step {step + 1}/{num_steps}: Forward-backward...")

        fwdbwd_response = requests.post(
            f"{BASE_URL}/fwdbwd",
            json={
                "model_id": model_id,
                "data": training_data,
                "loss_fn": "cross_entropy"
            }
        )

        future_id = fwdbwd_response.json()["request_id"]

        # Wait for forward-backward completion
        while True:
            result = requests.post(
                f"{BASE_URL}/retrieve_future",
                json={"request_id": future_id}
            )

            if result.status_code == 200:
                fwdbwd_result = result.json()
                loss = fwdbwd_result.get("loss", 0.0)
                print(f"[User {user_id}] Step {step + 1}: Loss = {loss:.4f}")
                break
            elif result.status_code == 202:
                time.sleep(0.1)
            else:
                raise Exception(f"Forward-backward failed: {result.text}")

        # Optimizer step
        print(f"[User {user_id}] Step {step + 1}/{num_steps}: Optimizer step...")

        optim_response = requests.post(
            f"{BASE_URL}/optim_step",
            json={
                "model_id": model_id,
                "adam_params": {
                    "learning_rate": 0.001
                }
            }
        )

        future_id = optim_response.json()["request_id"]

        # Wait for optimizer step completion
        while True:
            result = requests.post(
                f"{BASE_URL}/retrieve_future",
                json={"request_id": future_id}
            )

            if result.status_code == 200:
                optim_result = result.json()
                print(f"[User {user_id}] Step {step + 1}: Optimizer step completed")
                metrics.append({"step": step + 1, "loss": loss})
                break
            elif result.status_code == 202:
                time.sleep(0.1)
            else:
                raise Exception(f"Optimizer step failed: {result.text}")

    print(f"[User {user_id}] Training completed!")
    return {"user_id": user_id, "model_id": model_id, "metrics": metrics}


def test_inference(user_id: str, model_id: str) -> str:
    """Test inference with the trained adapter."""
    print(f"[User {user_id}] Testing inference...")

    response = requests.post(
        f"{BASE_URL}/api/v1/sample",
        json={
            "model_id": model_id,
            "prompts": [f"Hello from user {user_id}!"],
            "sampling_params": {
                "max_tokens": 20,
                "temperature": 0.7
            }
        }
    )

    if response.status_code == 200:
        generated_text = response.json().get("generated_text", "")
        print(f"[User {user_id}] Generated: {generated_text}")
        return generated_text
    else:
        raise Exception(f"Inference failed: {response.text}")


def simulate_user(user_id: str) -> Dict:
    """Simulate a complete user workflow: create adapter, train, test inference."""
    start_time = time.time()

    try:
        # Create adapter
        model_id = create_lora_adapter(user_id)

        # Train adapter
        training_results = train_adapter(user_id, model_id, num_steps=3)

        # Test inference
        generated_text = test_inference(user_id, model_id)

        elapsed = time.time() - start_time

        return {
            "user_id": user_id,
            "success": True,
            "model_id": model_id,
            "training_results": training_results,
            "generated_text": generated_text,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[User {user_id}] ERROR: {e}")
        return {
            "user_id": user_id,
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed
        }


def test_concurrent_users(num_users: int = 4):
    """Test concurrent users training different LoRA adapters."""
    print(f"\n{'='*80}")
    print(f"Testing {num_users} concurrent users with different LoRA adapters")
    print(f"{'='*80}\n")

    # Check server health
    try:
        health = requests.get(f"{BASE_URL}/healthz")
        if health.status_code != 200:
            print(f"ERROR: Server not healthy: {health.text}")
            return
        print("✓ Server is healthy\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        return

    start_time = time.time()

    # Run users concurrently
    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = {
            executor.submit(simulate_user, f"user_{i}"): f"user_{i}"
            for i in range(num_users)
        }

        results = []
        for future in as_completed(futures):
            user_id = futures[future]
            result = future.result()
            results.append(result)

    total_elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}\n")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total users: {num_users}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total elapsed time: {total_elapsed:.2f}s")

    if successful:
        avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
        print(f"Average time per user: {avg_time:.2f}s")

    print("\nPer-user details:")
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {result['user_id']}: {result['elapsed_time']:.2f}s", end="")
        if result["success"]:
            print(f" (model: {result['model_id']})")
        else:
            print(f" (error: {result.get('error', 'unknown')})")

    # Verify concurrent execution benefit
    if len(successful) > 1:
        sequential_time_estimate = sum(r["elapsed_time"] for r in successful)
        speedup = sequential_time_estimate / total_elapsed
        print(f"\nConcurrency speedup: {speedup:.2f}x")
        print(f"(Estimated sequential time: {sequential_time_estimate:.2f}s vs actual: {total_elapsed:.2f}s)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys

    num_users = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    test_concurrent_users(num_users)
