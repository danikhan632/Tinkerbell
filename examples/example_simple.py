"""
Simple example using the Flask server with requests library.

This shows how to use the server without the Tinker SDK.
"""

import requests
import time

BASE_URL = "http://localhost:8000"


def wait_for_result(future_id: str, timeout: int = 30) -> dict:
    """Poll for async job result."""
    elapsed = 0
    while elapsed < timeout:
        response = requests.post(f"{BASE_URL}/retrieve_future", json={
            "request_id": future_id
        })

        if response.status_code == 200:
            return response.json()

        time.sleep(0.5)
        elapsed += 0.5

    raise TimeoutError(f"Job {future_id} did not complete within {timeout} seconds")


def main():
    print("=== Flask Server Example ===\n")

    # Check server health
    health = requests.get(f"{BASE_URL}/healthz").json()
    print(f"Server status: {health['status']}\n")

    # List available loss functions
    loss_fns = requests.get(f"{BASE_URL}/loss_functions").json()
    print("Available loss functions:")
    for fn in loss_fns.get("available_loss_functions", []):
        desc = loss_fns.get("descriptions", {}).get(fn, "")
        print(f"  - {fn}: {desc}")
    print()

    # Training data
    training_data = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ],
        [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ]
    ]

    print("=== Example 1: Supervised Learning (cross_entropy) ===\n")

    model_id = "supervised-model"

    for step in range(3):
        print(f"--- Step {step + 1} ---")

        # Submit forward-backward job
        fwdbwd_response = requests.post(f"{BASE_URL}/fwdbwd", json={
            "model_id": model_id,
            "data": training_data,
            "loss_fn": "cross_entropy"
        })

        future_id = fwdbwd_response.json()["request_id"]
        print(f"Submitted forward-backward job: {future_id}")

        # Wait for result
        fwdbwd_result = wait_for_result(future_id)
        print(f"Loss: {fwdbwd_result.get('loss', 'N/A'):.4f}")

        metrics = fwdbwd_result.get('metrics', {})
        if metrics:
            print(f"Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.4f}")
                else:
                    print(f"  - {key}: {value}")

        # Submit optimizer step
        optim_response = requests.post(f"{BASE_URL}/optim_step", json={
            "model_id": model_id,
            "adam_params": {"learning_rate": 5e-4}
        })

        optim_future_id = optim_response.json()["request_id"]
        optim_result = wait_for_result(optim_future_id)
        print(f"Optimizer step complete\n")

    print("\n=== Example 2: RL Training (PPO) ===\n")

    rl_model_id = "rl-model-ppo"

    # RL training data
    rl_data = [
        [
            {"role": "user", "content": "What is 10 - 3?"},
            {"role": "assistant", "content": "10 - 3 = 7"}
        ]
    ]

    # Simulated RL inputs
    target_tokens = [[220, 16, 15, 489, 220, 18, 284, 220, 22]]
    sampling_logprobs = [[-0.4, -0.5, -0.3, -0.4, -0.3, -0.5, -0.2, -0.4, -0.3]]
    advantages = [[0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0]]

    for step in range(3):
        print(f"--- Step {step + 1} ---")

        # Submit forward-backward with PPO
        fwdbwd_response = requests.post(f"{BASE_URL}/fwdbwd", json={
            "model_id": rl_model_id,
            "data": rl_data,
            "loss_fn": "ppo",
            "loss_fn_inputs": {
                "target_tokens": target_tokens,
                "logprobs": sampling_logprobs,
                "advantages": advantages
            }
        })

        future_id = fwdbwd_response.json()["request_id"]
        fwdbwd_result = wait_for_result(future_id)

        print(f"Loss: {fwdbwd_result.get('loss', 'N/A'):.4f}")

        metrics = fwdbwd_result.get('metrics', {})
        if metrics:
            print(f"PPO Metrics:")
            print(f"  - Mean advantage: {metrics.get('mean_advantage', 0):.4f}")
            print(f"  - Mean ratio: {metrics.get('mean_ratio', 0):.4f}")
            print(f"  - Clip fraction: {metrics.get('clip_fraction', 0):.4f}")

        # Optimizer step
        optim_response = requests.post(f"{BASE_URL}/optim_step", json={
            "model_id": rl_model_id,
            "adam_params": {"learning_rate": 5e-4}
        })

        optim_future_id = optim_response.json()["request_id"]
        wait_for_result(optim_future_id)
        print(f"Optimizer step complete\n")

    print("\n=== Example 3: List Adapters ===\n")

    # List all trained adapters
    adapters = requests.get(f"{BASE_URL}/adapters").json()
    print(f"Trained adapters ({len(adapters.get('adapters', []))}):")
    for adapter in adapters.get('adapters', []):
        print(f"  - {adapter['model_id']}: {adapter['trainable_params']:,} params, "
              f"gradients={adapter['has_gradients']}")

    print("\n=== Training Complete! ===")
    print(f"\nYou now have {len(adapters.get('adapters', []))} trained models:")
    print("  1. 'supervised-model' - Trained with cross_entropy")
    print("  2. 'rl-model-ppo' - Trained with PPO")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server at http://localhost:8000")
        print("Make sure the server is running:")
        print("  cd /home/green/code/thinker/flask_server")
        print("  python app.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
