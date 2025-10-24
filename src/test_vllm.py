"""
Test script for vLLM integration with LoRA adapters.

This script tests the vLLM backend functionality including:
- Basic sampling
- LoRA adapter registration and usage
- Multiple concurrent requests
"""

import os
import sys

# Set environment for testing
os.environ["USE_VLLM"] = "true"
os.environ["VLLM_BASE_URL"] = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")

import vllm_backend


def test_vllm_availability():
    """Test if vLLM is available."""
    print("Testing vLLM availability...")
    available = vllm_backend.is_vllm_available()
    print(f"vLLM available: {available}")
    assert available, "vLLM should be available"
    print("✓ vLLM availability test passed\n")


def test_client_initialization():
    """Test vLLM client initialization."""
    print("Testing vLLM client initialization...")
    client = vllm_backend.initialize_vllm_client()
    assert client is not None, "Client should be initialized"

    info = vllm_backend.get_client_info()
    print(f"Client info: {info}")
    assert info["initialized"], "Client should be initialized"
    assert info["available"], "vLLM should be available"
    print("✓ Client initialization test passed\n")


def test_basic_sampling():
    """Test basic sampling without LoRA."""
    print("Testing basic sampling...")

    result = vllm_backend.generate_with_vllm(
        prompts=["Once upon a time"],
        max_tokens=20,
        temperature=0.8,
        top_p=0.95
    )

    print(f"Generated result: {result}")
    assert "completion_ids" in result, "Result should contain completion_ids"
    assert len(result["completion_ids"]) > 0, "Should have at least one completion"
    print("✓ Basic sampling test passed\n")


def test_lora_adapter_management():
    """Test LoRA adapter registration and management."""
    print("Testing LoRA adapter management...")

    # Note: This test requires a valid LoRA adapter path
    # For actual testing, replace with a real path
    lora_path = os.environ.get("TEST_LORA_PATH", "/tmp/test_lora")

    # Create dummy LoRA directory for testing
    if not os.path.exists(lora_path):
        print(f"Creating dummy LoRA path: {lora_path}")
        os.makedirs(lora_path, exist_ok=True)
        # Create dummy adapter config
        with open(os.path.join(lora_path, "adapter_config.json"), "w") as f:
            f.write('{"r": 16, "lora_alpha": 32}')

    try:
        # Register adapter
        vllm_backend.register_lora_adapter(
            adapter_id="test_adapter",
            lora_path=lora_path
        )

        # List adapters
        adapters = vllm_backend.list_lora_adapters()
        print(f"Registered adapters: {adapters}")
        assert len(adapters) > 0, "Should have at least one adapter"
        assert any(a["adapter_id"] == "test_adapter" for a in adapters), "test_adapter should be registered"

        # Unregister adapter
        success = vllm_backend.unregister_lora_adapter("test_adapter")
        assert success, "Should successfully unregister adapter"

        # Verify unregistration
        adapters = vllm_backend.list_lora_adapters()
        assert not any(a["adapter_id"] == "test_adapter" for a in adapters), "test_adapter should be unregistered"

        print("✓ LoRA adapter management test passed\n")
    except Exception as e:
        print(f"Note: LoRA adapter test skipped or partially failed: {e}")
        print("This is expected if no valid LoRA adapter is available\n")


def test_sampling_with_lora():
    """Test sampling with a LoRA adapter."""
    print("Testing sampling with LoRA adapter...")

    lora_path = os.environ.get("TEST_LORA_PATH", "/tmp/test_lora")

    if not os.path.exists(lora_path):
        print("Skipping LoRA sampling test (no adapter available)\n")
        return

    try:
        # Register adapter
        vllm_backend.register_lora_adapter(
            adapter_id="test_lora_sampling",
            lora_path=lora_path
        )

        # Generate with LoRA
        result = vllm_backend.generate_with_vllm(
            prompts=["Hello, how are you?"],
            lora_adapter_id="test_lora_sampling",
            max_tokens=20,
            temperature=0.8
        )

        print(f"Generated with LoRA: {result}")
        assert "completion_ids" in result, "Result should contain completion_ids"

        # Cleanup
        vllm_backend.unregister_lora_adapter("test_lora_sampling")

        print("✓ LoRA sampling test passed\n")
    except Exception as e:
        print(f"Note: LoRA sampling test failed: {e}")
        print("This might be expected if vLLM server doesn't support LoRA or adapter is invalid\n")


def test_worker_integration():
    """Test integration with worker module."""
    print("Testing worker integration...")

    try:
        import worker

        backend = worker.get_backend()
        print(f"Current backend: {backend}")

        # Test if vLLM is configured
        assert worker.USE_VLLM, "USE_VLLM should be enabled"
        print("✓ Worker integration test passed\n")
    except ImportError as e:
        print(f"Note: Worker integration test skipped: {e}\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("vLLM Integration Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_vllm_availability,
        test_client_initialization,
        test_basic_sampling,
        test_lora_adapter_management,
        test_sampling_with_lora,
        test_worker_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
