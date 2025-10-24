"""
Example client for the Tinker API Server.

Usage:
    python client_example.py
"""
import requests
import time
from typing import Optional, Dict, Any


class TinkerClient:
    """Client for interacting with the Tinker API Server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> dict:
        """Check server health."""
        response = requests.get(f"{self.base_url}/healthz")
        response.raise_for_status()
        return response.json()

    def get_capabilities(self) -> dict:
        """Get server capabilities and supported models."""
        response = requests.get(f"{self.base_url}/get_server_capabilities")
        response.raise_for_status()
        return response.json()

    def sample(self, prompts: list[str], model_id: str = None, base_model: str = None, **kwargs) -> dict:
        """Synchronous sampling (blocks until complete).

        Args:
            prompts: List of prompts to sample from
            model_id: Model/adapter ID to use (for trained adapters)
            base_model: Base model to use (if not using a trained adapter)
            **kwargs: Additional sampling parameters
        """
        data = {
            "prompts": prompts,
            **kwargs
        }

        # Add model_id or base_model
        if model_id:
            data["model_id"] = model_id
        elif base_model:
            data["base_model"] = base_model
        else:
            data["base_model"] = "base"  # Default to base model

        response = requests.post(f"{self.base_url}/api/v1/sample", json=data)
        response.raise_for_status()
        return response.json()

    def asample(self, prompts: list[str], model_id: str = None, base_model: str = None, **kwargs) -> str:
        """Asynchronous sampling (returns request_id for polling).

        Args:
            prompts: List of prompts to sample from
            model_id: Model/adapter ID to use (for trained adapters)
            base_model: Base model to use (if not using a trained adapter)
            **kwargs: Additional sampling parameters
        """
        data = {
            "prompts": prompts,
            **kwargs
        }

        # Add model_id or base_model
        if model_id:
            data["model_id"] = model_id
        elif base_model:
            data["base_model"] = base_model
        else:
            data["base_model"] = "base"  # Default to base model

        response = requests.post(f"{self.base_url}/api/v1/asample", json=data)
        response.raise_for_status()
        return response.json()["request_id"]

    def forward(self, model_id: str, data: list, **kwargs) -> str:
        """Forward pass - returns request_id."""
        params = {
            "model_id": model_id,
            "data": data,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/fwd", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def fwdbwd(self, model_id: str, data: list, loss_fn: str = "cross_entropy", **kwargs) -> str:
        """Forward-backward pass - returns request_id."""
        params = {
            "model_id": model_id,
            "data": data,
            "loss_fn": loss_fn,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/fwdbwd", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def optim_step(self, model_id: str, **kwargs) -> str:
        """Optimizer step - returns request_id."""
        params = {
            "model_id": model_id,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/optim_step", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def retrieve_future(self, request_id: str) -> dict:
        """Retrieve result of an async operation."""
        data = {"request_id": request_id}
        response = requests.post(f"{self.base_url}/retrieve_future", json=data)
        response.raise_for_status()
        return response.json()

    def wait_for_result(self, request_id: str, timeout: int = 60, poll_interval: float = 0.5) -> dict:
        """Poll for result until complete or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.post(
                    f"{self.base_url}/retrieve_future",
                    json={"request_id": request_id}
                )

                # 202 = still pending, continue polling
                if response.status_code == 202:
                    time.sleep(poll_interval)
                    continue

                # 500 = error occurred
                if response.status_code == 500:
                    try:
                        error_detail = response.json().get('detail', 'Unknown error')
                    except:
                        error_detail = response.text
                    raise RuntimeError(f"Job failed: {error_detail}")

                # 200 = success, return the result
                if response.status_code == 200:
                    return response.json()

                # Other error
                response.raise_for_status()

            except requests.RequestException as e:
                if time.time() - start_time >= timeout:
                    raise TimeoutError(f"Request {request_id} timed out after {timeout}s")
                # Continue on connection errors (server might be busy)
                time.sleep(poll_interval)
                continue

        raise TimeoutError(f"Request {request_id} timed out after {timeout}s")

    def add_lora(self, base_model: str = "base", rank: int = 16, alpha: int = 32, **kwargs) -> str:
        """Add LoRA adapter - returns request_id."""
        params = {
            "base_model": base_model,
            "rank": rank,
            "alpha": alpha,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/add_lora", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def remove_lora(self, model_id: str, **kwargs) -> str:
        """Remove LoRA adapter - returns request_id."""
        params = {
            "model_id": model_id,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/remove_lora", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def save_weights(self, model_id: str, path: str, **kwargs) -> str:
        """Save model weights - returns request_id."""
        params = {
            "model_id": model_id,
            "path": path,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/save_weights", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def load_weights(self, model_id: str, path: str, **kwargs) -> str:
        """Load model weights - returns request_id."""
        params = {
            "model_id": model_id,
            "path": path,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/load_weights", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def save_weights_for_sampler(self, model_id: str, path: str, **kwargs) -> str:
        """Save weights for sampler - returns request_id."""
        params = {
            "model_id": model_id,
            "path": path,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/save_weights_for_sampler", json=params)
        response.raise_for_status()
        return response.json()["request_id"]

    def get_info(self, model_id: str) -> dict:
        """Get model info."""
        data = {"model_id": model_id}
        response = requests.post(f"{self.base_url}/get_info", json=data)
        response.raise_for_status()
        return response.json()


def main():
    """Run example client operations."""
    client = TinkerClient("http://localhost:8000")

    print("=" * 60)
    print("Tinker API Client Example")
    print("=" * 60)

    # 1. Health check
    print("\n[1] Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"Status: {health}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Get capabilities
    print("\n[2] Server Capabilities")
    print("-" * 60)
    try:
        capabilities = client.get_capabilities()
        print(f"Supported models: {capabilities.get('supported_models', [])}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Forward-backward training example
    print("\n[3] Training Example (Forward-Backward)")
    print("-" * 60)
    training_data = [[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]]

    try:
        request_id = client.fwdbwd(
            model_id="llama-3-8b",
            data=training_data,
            loss_fn="cross_entropy"
        )
        print(f"Request ID: {request_id}")
        print("Waiting for result...")

        result = client.wait_for_result(request_id, timeout=60)
        print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # 4. Forward pass example
    print("\n[4] Forward Pass Example")
    print("-" * 60)
    forward_data = [[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]]

    try:
        request_id = client.forward(
            model_id="llama-3-8b",
            data=forward_data
        )
        print(f"Request ID: {request_id}")
        print("Waiting for result...")

        result = client.wait_for_result(request_id, timeout=60)
        print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # 5. Optimizer step example
    print("\n[5] Optimizer Step Example")
    print("-" * 60)
    try:
        request_id = client.optim_step(
            model_id="llama-3-8b"
        )
        print(f"Request ID: {request_id}")
        print("Waiting for result...")

        result = client.wait_for_result(request_id, timeout=60)
        print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # 6. Async sampling example
    print("\n[6] Async Sampling Example")
    print("-" * 60)
    try:
        request_id = client.asample(
            prompts=["Tell me a short joke about programming."],
            base_model="base",
            sampling_params={"max_tokens": 100}
        )
        print(f"Request ID: {request_id}")
        print("Waiting for result...")

        result = client.wait_for_result(request_id, timeout=60)
        print(f"Result: {result}")
    except TimeoutError as e:
        print(f"Timeout: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # 7. Synchronous sampling example
    print("\n[7] Synchronous Sampling Example")
    print("-" * 60)
    try:
        result = client.sample(
            prompts=["What is Python?"],
            base_model="base",
            sampling_params={"max_tokens": 50}
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # 8. LoRA management example
    print("\n[8] LoRA Management Example")
    print("-" * 60)
    try:
        # Add LoRA
        print("Creating LoRA adapter...")
        request_id = client.add_lora(
            base_model="base",
            rank=8,
            alpha=16
        )
        print(f"Add LoRA Request ID: {request_id}")
        print("Waiting for adapter creation...")

        result = client.wait_for_result(request_id, timeout=30)
        adapter_model_id = result.get("model_id", "unknown")
        print(f"✓ Adapter created: {adapter_model_id}")

        # Train the adapter
        print("\nTraining the adapter...")
        training_data = [[
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]]

        request_id = client.fwdbwd(
            model_id=adapter_model_id,
            data=training_data,
            loss_fn="cross_entropy"
        )
        print(f"Training Request ID: {request_id}")
        result = client.wait_for_result(request_id, timeout=30)
        print(f"✓ Training result: {result}")

        # Remove LoRA (cleanup)
        print("\nRemoving adapter...")
        request_id = client.remove_lora(model_id=adapter_model_id)
        print(f"Remove LoRA Request ID: {request_id}")
        result = client.wait_for_result(request_id, timeout=30)
        print(f"✓ Adapter removed: {result}")

    except Exception as e:
        print(f"Error: {e}")

    # 9. Model info example
    print("\n[9] Model Info Example")
    print("-" * 60)
    try:
        info = client.get_info(model_id="llama-3-8b")
        print(f"Model Info: {info}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
