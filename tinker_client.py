"""
Tinkerbell Client Library

Provides decorators and utilities for users to define custom loss functions
that are automatically uploaded to the Tinkerbell server.

Example usage:
    from tinker_client import custom_loss, TinkerClient

    client = TinkerClient("http://localhost:8000")

    @custom_loss(client, name="my_dpo_loss")
    def dpo_loss(model_outputs, loss_fn_inputs, attention_mask, beta=0.1):
        # Your custom loss implementation
        ...
        return LossFnOutput(loss=loss, logprobs=logprobs, diagnostics={})
"""

import requests
import time
from typing import Callable, Optional, Dict, Any
from functools import wraps


class TinkerClient:
    """
    Client for interacting with Tinkerbell server.

    Args:
        base_url: URL of the Tinkerbell server (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._registered_losses = set()

    def register_custom_loss(self, name: str, loss_fn: Callable) -> Dict[str, Any]:
        """
        Register a custom loss function with the server.

        Args:
            name: Name to register the loss function under
            loss_fn: The loss function to register

        Returns:
            Response dict from server

        Raises:
            RuntimeError: If registration fails
        """
        try:
            import cloudpickle
            import base64

            # Serialize the function
            loss_fn_bytes = cloudpickle.dumps(loss_fn)
            loss_fn_serialized = base64.b64encode(loss_fn_bytes).decode('utf-8')

            # Send to server
            response = requests.post(
                f"{self.base_url}/register_custom_loss",
                json={
                    "loss_name": name,
                    "loss_fn_serialized": loss_fn_serialized
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise RuntimeError(f"Failed to register loss function: {response.text}")

            result = response.json()
            request_id = result.get("request_id")

            # Poll for completion
            max_wait = 10  # seconds
            elapsed = 0
            while elapsed < max_wait:
                retrieve_response = requests.post(
                    f"{self.base_url}/retrieve_future",
                    json={"request_id": request_id},
                    timeout=self.timeout
                )

                if retrieve_response.status_code == 200:
                    registration_result = retrieve_response.json()
                    self._registered_losses.add(name)
                    print(f"✓ Custom loss '{name}' registered successfully")
                    return registration_result
                elif retrieve_response.status_code == 202:
                    # Still pending
                    time.sleep(0.5)
                    elapsed += 0.5
                else:
                    raise RuntimeError(f"Registration failed: {retrieve_response.text}")

            raise RuntimeError(f"Registration timed out after {max_wait}s")

        except ImportError:
            raise RuntimeError(
                "cloudpickle is required for custom loss functions. "
                "Install with: pip install cloudpickle"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to register custom loss '{name}': {e}")

    def list_loss_functions(self) -> Dict[str, Any]:
        """
        List all available loss functions on the server.

        Returns:
            Dict containing available loss functions and descriptions
        """
        response = requests.get(
            f"{self.base_url}/list_loss_functions",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def healthz(self) -> Dict[str, str]:
        """Check server health."""
        response = requests.get(f"{self.base_url}/healthz", timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def custom_loss(
    client: TinkerClient,
    name: Optional[str] = None
) -> Callable:
    """
    Decorator to register a custom loss function with the Tinkerbell server.

    This decorator serializes the function and uploads it to the server,
    making it available for use in training.

    Args:
        client: TinkerClient instance connected to the server
        name: Name to register the loss function under (defaults to function name)

    Example:
        @custom_loss(client, name="my_dpo_loss")
        def dpo_loss(model_outputs, loss_fn_inputs, attention_mask, beta=0.1):
            # Your loss implementation
            ...
            return LossFnOutput(loss=loss, logprobs=logprobs, diagnostics={})

        # Now you can use "my_dpo_loss" in training:
        # forward_backward(data=data, loss_fn="my_dpo_loss", ...)

    Returns:
        Decorated function (unchanged, for local use if needed)
    """

    def decorator(func: Callable) -> Callable:
        loss_name = name or func.__name__

        # Register the function with the server
        client.register_custom_loss(loss_name, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Function still works locally if needed
            return func(*args, **kwargs)

        # Attach metadata
        wrapper._registered_name = loss_name
        wrapper._client = client

        return wrapper

    return decorator


# Convenience function for creating LossFnOutput
def LossFnOutput(loss, logprobs, diagnostics):
    """
    Create a LossFnOutput dataclass-like object.

    This is a helper for users writing custom losses to match the expected return format.

    Args:
        loss: Scalar loss tensor
        logprobs: Log probabilities tensor
        diagnostics: Dict of diagnostic metrics

    Returns:
        Object with loss, logprobs, and diagnostics attributes
    """
    from dataclasses import dataclass
    import torch

    @dataclass
    class _LossFnOutput:
        loss: torch.Tensor
        logprobs: torch.Tensor
        diagnostics: Dict[str, float]

    return _LossFnOutput(loss=loss, logprobs=logprobs, diagnostics=diagnostics)
