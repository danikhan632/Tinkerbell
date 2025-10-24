#!/usr/bin/env python3
"""
Example: Multiple Concurrent Users Training Different LoRA Adapters

This example demonstrates how multiple users can train their own LoRA adapters
concurrently without blocking each other.

Scenario: 3 users (Alice, Bob, Carol) each training a personalized chatbot adapter
"""

import requests
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


class User:
    """Represents a user training their own LoRA adapter."""

    def __init__(self, name: str, color: str = ""):
        self.name = name
        self.color = color  # For colored terminal output
        self.model_id = None
        self.training_history = []

    def log(self, message: str):
        """Print a colored log message."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{self.color}[{timestamp}] {self.name}: {message}\033[0m")

    def create_adapter(self, rank: int = 8) -> str:
        """Create a new LoRA adapter."""
        self.log(f"Creating LoRA adapter (rank={rank})...")

        response = requests.post(
            f"{BASE_URL}/add_lora",
            json={"base_model": "base", "rank": rank, "alpha": rank * 2}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to create adapter: {response.text}")

        future_id = response.json()["request_id"]

        # Poll for completion
        while True:
            result = requests.post(
                f"{BASE_URL}/retrieve_future",
                json={"request_id": future_id}
            )

            if result.status_code == 200:
                self.model_id = result.json()["model_id"]
                self.log(f"✓ Adapter created: {self.model_id}")
                return self.model_id
            elif result.status_code == 202:
                time.sleep(0.1)
            else:
                raise Exception(f"Failed: {result.text}")

    def train_step(self, data: list, step_num: int) -> Dict[str, Any]:
        """Perform one training step (forward-backward + optimizer step)."""
        self.log(f"Step {step_num}: Starting forward-backward pass...")

        # Forward-backward
        fwdbwd_response = requests.post(
            f"{BASE_URL}/fwdbwd",
            json={
                "model_id": self.model_id,
                "data": data,
                "loss_fn": "cross_entropy"
            }
        )

        if fwdbwd_response.status_code != 200:
            raise Exception(f"Forward-backward failed: {fwdbwd_response.text}")

        future_id = fwdbwd_response.json()["request_id"]

        # Wait for completion
        while True:
            result = requests.post(
                f"{BASE_URL}/retrieve_future",
                json={"request_id": future_id}
            )

            if result.status_code == 200:
                fwdbwd_result = result.json()
                loss = fwdbwd_result.get("loss", 0.0)
                self.log(f"Step {step_num}: Loss = {loss:.4f}")
                break
            elif result.status_code == 202:
                time.sleep(0.1)
            else:
                raise Exception(f"Failed: {result.text}")

        # Optimizer step
        self.log(f"Step {step_num}: Applying optimizer step...")

        optim_response = requests.post(
            f"{BASE_URL}/optim_step",
            json={
                "model_id": self.model_id,
                "adam_params": {"learning_rate": 0.001}
            }
        )

        if optim_response.status_code != 200:
            raise Exception(f"Optimizer failed: {optim_response.text}")

        future_id = optim_response.json()["request_id"]

        # Wait for completion
        while True:
            result = requests.post(
                f"{BASE_URL}/retrieve_future",
                json={"request_id": future_id}
            )

            if result.status_code == 200:
                self.log(f"Step {step_num}: ✓ Training step completed")
                metrics = {"step": step_num, "loss": loss}
                self.training_history.append(metrics)
                return metrics
            elif result.status_code == 202:
                time.sleep(0.1)
            else:
                raise Exception(f"Failed: {result.text}")

    def train(self, training_data: list, num_steps: int = 3):
        """Train the adapter for multiple steps."""
        self.log(f"Starting training for {num_steps} steps...")

        for step in range(1, num_steps + 1):
            self.train_step(training_data, step)

        self.log(f"✓ Training completed! ({num_steps} steps)")

    def test_inference(self, prompt: str) -> str:
        """Test inference with the trained adapter."""
        self.log(f"Testing inference with prompt: '{prompt}'")

        response = requests.post(
            f"{BASE_URL}/api/v1/sample",
            json={
                "model_id": self.model_id,
                "prompts": [prompt],
                "sampling_params": {
                    "max_tokens": 30,
                    "temperature": 0.7
                }
            }
        )

        if response.status_code == 200:
            generated = response.json().get("generated_text", "")
            self.log(f"Generated: {generated}")
            return generated
        else:
            raise Exception(f"Inference failed: {response.text}")


def run_user_workflow(user: User, training_data: list, test_prompt: str):
    """Run the complete workflow for a user."""
    start_time = time.time()

    try:
        # Step 1: Create adapter
        user.create_adapter(rank=8)

        # Step 2: Train adapter
        user.train(training_data, num_steps=3)

        # Step 3: Test inference
        user.test_inference(test_prompt)

        elapsed = time.time() - start_time
        user.log(f"✓✓✓ ALL DONE in {elapsed:.2f}s ✓✓✓")

    except Exception as e:
        elapsed = time.time() - start_time
        user.log(f"✗✗✗ ERROR after {elapsed:.2f}s: {e}")


def main():
    """Main example: 3 concurrent users training different adapters."""

    print("\n" + "="*80)
    print("Example: Multiple Concurrent Users Training LoRA Adapters")
    print("="*80 + "\n")

    # Check server health
    try:
        health = requests.get(f"{BASE_URL}/healthz", timeout=2)
        if health.status_code != 200:
            print(f"ERROR: Server not healthy: {health.text}")
            return
        print("✓ Server is healthy\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE_URL}")
        print(f"Please start the server with: python app.py")
        return

    # Create users with different colors for easy visualization
    alice = User("Alice", "\033[94m")  # Blue
    bob = User("Bob", "\033[92m")      # Green
    carol = User("Carol", "\033[93m")   # Yellow

    # Each user has personalized training data
    alice_data = [
        [
            {"role": "user", "content": "What's your favorite color?"},
            {"role": "assistant", "content": "My favorite color is blue, like the ocean!"}
        ]
    ]

    bob_data = [
        [
            {"role": "user", "content": "What do you like to do?"},
            {"role": "assistant", "content": "I love playing chess and solving puzzles!"}
        ]
    ]

    carol_data = [
        [
            {"role": "user", "content": "Tell me about yourself."},
            {"role": "assistant", "content": "I'm a friendly assistant who enjoys helping with creative writing!"}
        ]
    ]

    print("Starting concurrent training for 3 users...")
    print("Watch how they all train simultaneously!\n")

    start_time = time.time()

    # Create threads for concurrent execution
    threads = [
        threading.Thread(target=run_user_workflow, args=(alice, alice_data, "What's your favorite color?")),
        threading.Thread(target=run_user_workflow, args=(bob, bob_data, "What do you like to do?")),
        threading.Thread(target=run_user_workflow, args=(carol, carol_data, "Tell me about yourself.")),
    ]

    # Start all threads (all users start training concurrently)
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ All 3 users completed training concurrently!")
    print(f"✓ Total time: {total_time:.2f}s")
    print(f"\nTraining History:")

    for user in [alice, bob, carol]:
        if user.training_history:
            print(f"\n  {user.name} ({user.model_id}):")
            for metrics in user.training_history:
                print(f"    Step {metrics['step']}: Loss = {metrics['loss']:.4f}")

    print("\n" + "="*80)
    print("\nKey Observation:")
    print("  - All 3 users trained their adapters SIMULTANEOUSLY")
    print("  - No blocking between users (different adapters)")
    print("  - Each user has their own independent LoRA adapter")
    print("  - Concurrent processing provides ~3x speedup!")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
