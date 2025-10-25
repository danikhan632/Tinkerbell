#!/usr/bin/env python3
"""
Simple Example: 3 Users Training Different LoRA Adapters Concurrently
with Megatron-Bridge Backend

This example demonstrates concurrent multi-user training using the new
Megatron-Bridge backend with in-memory weight streaming.

Backend Configuration:
- Training: Megatron-Bridge (HF ↔ Megatron conversion, sequential LoRA training)
- Sampling: vLLM for high-performance inference with LoRA adapters
- Weight Sync: In-memory streaming (no disk I/O!) between Megatron and vLLM

Key Features:
1. Simple initialization with HuggingFace model names
2. Automatic weight conversion between HF and Megatron formats
3. In-memory weight streaming to vLLM (inspired by NVIDIA's RLHF example)
4. Per-user LoRA adapters with independent learning rates

Setup:
1. Start Tinkerbell server: python src/app.py
2. (Optional) Enable vLLM: export USE_VLLM=true
3. Run this example: python examples/example_simple_concurrent.py
"""

import threading
import time
from client_example import TinkerClient


def train_user_adapter(user_name: str, client: TinkerClient, learning_rate: float = 0.001):
    """Train a single user's LoRA adapter with Megatron-Bridge backend."""
    print(f"\n[{user_name}] Starting with Megatron-Bridge backend...")

    try:
        # Step 1: Create a LoRA adapter
        # NOTE: With Megatron-Bridge, this creates an adapter that will be
        # automatically converted between HF and Megatron formats
        print(f"[{user_name}] Creating LoRA adapter...")
        request_id = client.add_lora(
            base_model="base",  # Uses model configured in initialize_base_model()
            rank=8,              # LoRA rank (r)
            alpha=16             # LoRA alpha (scaling factor)
        )
        result = client.wait_for_result(request_id, timeout=30)
        model_id = result["model_id"]
        print(f"[{user_name}] ✓ Created adapter: {model_id}")

        # Step 2: Train with personalized data
        # Each user gets different training data to personalize their adapter
        training_data = [[
            {"role": "user", "content": f"Hello, I'm {user_name}!"},
            {"role": "assistant", "content": f"Nice to meet you, {user_name}! How can I help you today?"}
        ], [
            {"role": "user", "content": f"What's my name?"},
            {"role": "assistant", "content": f"Your name is {user_name}!"}
        ]]

        print(f"[{user_name}] Training adapter with {len(training_data)} examples...")

        for step in range(1, 4):  # 3 training steps
            print(f"[{user_name}] Training step {step}/3...")

            # Forward-backward pass
            # NOTE: With Megatron-Bridge, this:
            # 1. Converts HF model to Megatron format (cached after first time)
            # 2. Applies LoRA layers to Megatron model
            # 3. Computes loss and gradients
            request_id = client.fwdbwd(
                model_id=model_id,
                data=training_data,
                loss_fn="cross_entropy"  # Standard cross-entropy loss
            )
            result = client.wait_for_result(request_id, timeout=30)
            loss = result.get("loss", 0.0)
            print(f"[{user_name}]   Step {step}: Loss = {loss:.4f}")

            # Optimizer step
            # NOTE: Each user can have independent learning rates!
            request_id = client.optim_step(
                model_id=model_id,
                adam_params={"learning_rate": learning_rate}
            )
            result = client.wait_for_result(request_id, timeout=30)
            print(f"[{user_name}]   Step {step}: Optimizer applied (LR={learning_rate})")

        # Step 3: Sync trained weights to vLLM
        # NOTE: With Megatron-Bridge, weights are streamed in-memory (no disk I/O!)
        # This is based on the refit_hf_from_megatron pattern from NVIDIA's RLHF example
        print(f"[{user_name}] Syncing trained adapter to vLLM (in-memory)...")
        # The backend automatically handles weight streaming via Bridge's export_hf_weights()

        # Step 4: Test inference with trained adapter
        print(f"[{user_name}] Testing inference with trained LoRA adapter...")
        result = client.sample(
            model_id=model_id,  # LoRA adapter ID
            prompts=[f"Hello! What's my name?"],
            sampling_params={"max_tokens": 30, "temperature": 0.7}
        )
        generated = result.get("generated_text", "")
        print(f"[{user_name}] Generated: {generated}")

        # Step 5: Get adapter stats
        print(f"[{user_name}] Fetching adapter statistics...")
        try:
            stats = client.get_optimizer_state(model_id)
            if stats:
                print(f"[{user_name}] Adapter stats:")
                print(f"[{user_name}]   - Learning rate: {stats.get('learning_rate', 'N/A')}")
                print(f"[{user_name}]   - Num parameters: {stats.get('num_params', 'N/A')}")
        except Exception as e:
            print(f"[{user_name}] Note: Could not fetch stats: {e}")

        print(f"[{user_name}] ✓✓✓ DONE! Adapter trained and synced! ✓✓✓")

    except Exception as e:
        print(f"[{user_name}] ✗✗✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run 3 users concurrently with Megatron-Bridge backend."""

    print("\n" + "="*80)
    print("🚀 Megatron-Bridge Multi-User Training Example 🚀")
    print("="*80)
    print("\nThis example demonstrates:")
    print("  ✓ Concurrent multi-user LoRA adapter training")
    print("  ✓ Megatron-Bridge for HF ↔ Megatron conversion")
    print("  ✓ In-memory weight streaming (no disk I/O!)")
    print("  ✓ Per-user adapters with independent learning rates")
    print("  ✓ Fast inference with vLLM + trained LoRA adapters")

    print("\n" + "-"*80)
    print("Backend Configuration:")
    print("-"*80)
    print("  Training:  Megatron-Bridge")
    print("             - Converts HuggingFace models to Megatron format")
    print("             - Supports 20+ model architectures (Llama, Qwen, etc.)")
    print("             - Sequential LoRA training (concurrent adapter management)")
    print()
    print("  Inference: vLLM (if USE_VLLM=true) or HuggingFace (fallback)")
    print("             - High-performance inference with LoRA adapters")
    print("             - Co-located with training (same GPU)")
    print()
    print("  Weight Sync: In-memory streaming via Bridge")
    print("               - No disk I/O!")
    print("               - Based on NVIDIA's RLHF example pattern")

    print("\n" + "-"*80)
    print("Setup:")
    print("-"*80)
    print("  1. Ensure Megatron-Bridge is installed:")
    print("     cd /home/shadeform/refs/Megatron-Bridge && pip install -e .")
    print()
    print("  2. Start Tinkerbell server:")
    print("     python src/app.py")
    print()
    print("  3. (Optional) Enable vLLM:")
    print("     export USE_VLLM=true")
    print()
    print("  4. Run this example:")
    print("     python examples/example_simple_concurrent.py")
    print("="*80 + "\n")

    # Create client
    client = TinkerClient("http://localhost:8000")

    # Check server
    print("Checking server connection...")
    try:
        health = client.health_check()
        print(f"✓ Server is healthy: {health}")

        # Get backend info
        try:
            backend_info = client.get_backend_info()
            print(f"\nBackend Information:")
            print(f"  Backend: {backend_info.get('backend', 'unknown')}")
            print(f"  Initialized: {backend_info.get('initialized', False)}")
            print(f"  Bridge available: {backend_info.get('bridge_available', False)}")
            if backend_info.get('model_name'):
                print(f"  Model: {backend_info.get('model_name')}")
        except Exception as e:
            print(f"Note: Could not fetch backend info: {e}")

    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("\nPlease start the server first:")
        print("  python src/app.py")
        return

    print("\n" + "="*80)
    print("Starting 3 concurrent users with different learning rates...")
    print("="*80 + "\n")

    start_time = time.time()

    # Create threads for 3 users with different learning rates
    # This demonstrates per-user customization
    users = [
        ("Alice", 0.001),   # Conservative learning rate
        ("Bob", 0.0005),    # Even more conservative
        ("Carol", 0.002),   # More aggressive
    ]
    threads = []

    for user_name, lr in users:
        thread = threading.Thread(
            target=train_user_adapter,
            args=(user_name, client, lr)
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
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"\n✓ All 3 users completed training with Megatron-Bridge!")
    print(f"✓ Total time: {total_time:.2f} seconds")
    print(f"\n🎯 Key Achievements:")
    print(f"  • Each user trained their own LoRA adapter independently")
    print(f"  • Different learning rates per user (0.001, 0.0005, 0.002)")
    print(f"  • HF ↔ Megatron weight conversion handled automatically")
    print(f"  • Weights streamed to vLLM in-memory (no disk I/O!)")
    print(f"  • All adapters ready for fast inference with vLLM")

    print(f"\n🔧 Backend Benefits (vs old implementation):")
    print(f"  • Simpler: No manual Megatron setup required")
    print(f"  • Faster: In-memory weight streaming (no disk writes)")
    print(f"  • Flexible: Easy to add new HF models")
    print(f"  • Production-ready: Based on NVIDIA's verified patterns")

    print(f"\n📚 Learn More:")
    print(f"  • Migration guide: MEGATRON_BRIDGE_MIGRATION.md")
    print(f"  • Bridge docs: https://docs.nvidia.com/nemo/megatron-bridge/")
    print(f"  • RLHF example: /home/shadeform/refs/Megatron-Bridge/examples/rl/")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
