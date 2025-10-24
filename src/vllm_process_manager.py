"""
vLLM Process Manager - Manages vLLM server as a background process.

This module starts and manages a vLLM server subprocess, ensuring proper
cleanup on exit.
"""

import os
import sys
import time
import signal
import atexit
import subprocess
import requests
from typing import Optional


class VLLMProcessManager:
    """Manages a vLLM server subprocess."""

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        host: str = "0.0.0.0",
        port: int = 8001,
        enable_lora: bool = True,
        max_loras: int = 4,
        max_lora_rank: int = 64,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        extra_args: Optional[list] = None
    ):
        self.model = model
        self.host = host
        self.port = port
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

        # Register cleanup
        atexit.register(self.stop)

    def build_command(self) -> list:
        """Build the vLLM server command."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
        ]

        if self.enable_lora:
            cmd.extend([
                "--enable-lora",
                "--max-loras", str(self.max_loras),
                "--max-lora-rank", str(self.max_lora_rank),
            ])

        cmd.extend(self.extra_args)

        return cmd

    def start(self, startup_timeout: float = 120.0) -> bool:
        """
        Start the vLLM server process.

        Args:
            startup_timeout: Maximum time to wait for server startup (seconds)

        Returns:
            True if started successfully, False otherwise
        """
        if self.process is not None:
            print("vLLM process already running")
            return True

        print(f"Starting vLLM server: {self.model}")
        print(f"  Host: {self.host}:{self.port}")
        print(f"  LoRA enabled: {self.enable_lora}")
        if self.enable_lora:
            print(f"  Max LoRAs: {self.max_loras}, Max rank: {self.max_lora_rank}")

        cmd = self.build_command()
        print(f"  Command: {' '.join(cmd)}")

        try:
            # Start process with output redirected
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            print(f"vLLM process started (PID: {self.process.pid})")

            # Monitor startup
            return self._wait_for_startup(startup_timeout)

        except Exception as e:
            print(f"Failed to start vLLM process: {e}")
            self.process = None
            return False

    def _wait_for_startup(self, timeout: float) -> bool:
        """
        Wait for vLLM server to be ready.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if server is ready, False if timeout or error
        """
        health_url = f"{self.base_url}/health"
        start_time = time.time()
        last_output_time = start_time

        print("Waiting for vLLM server to be ready...")

        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                print("vLLM process terminated unexpectedly")
                return False

            # Read and print output (non-blocking)
            if self.process.stdout:
                import select
                if select.select([self.process.stdout], [], [], 0.1)[0]:
                    line = self.process.stdout.readline()
                    if line:
                        print(f"  [vLLM] {line.rstrip()}")
                        last_output_time = time.time()

            # Try health check
            try:
                response = requests.get(health_url, timeout=1.0)
                if response.status_code == 200:
                    print(f"✓ vLLM server is ready at {self.base_url}")
                    return True
            except requests.exceptions.RequestException:
                pass

            # Show progress every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"  Still waiting... ({int(elapsed)}s elapsed)")

            time.sleep(1)

        print(f"Timeout waiting for vLLM server after {timeout}s")
        return False

    def stop(self):
        """Stop the vLLM server process."""
        if self.process is None:
            return

        print(f"Stopping vLLM server (PID: {self.process.pid})...")

        try:
            # Try graceful shutdown first
            self.process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                self.process.wait(timeout=10.0)
                print("vLLM server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not stopped
                print("Force killing vLLM server...")
                self.process.kill()
                self.process.wait()
                print("vLLM server killed")

        except Exception as e:
            print(f"Error stopping vLLM process: {e}")

        finally:
            self.process = None

    def is_running(self) -> bool:
        """Check if vLLM process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_status(self) -> dict:
        """Get vLLM server status."""
        return {
            "running": self.is_running(),
            "pid": self.process.pid if self.process else None,
            "base_url": self.base_url,
            "model": self.model,
            "enable_lora": self.enable_lora,
        }


def create_from_env() -> Optional[VLLMProcessManager]:
    """
    Create vLLM process manager from environment variables.

    Environment Variables:
        VLLM_AUTO_START: Set to "true" to auto-start vLLM server
        VLLM_MODEL: Model to load (default: HuggingFaceTB/SmolLM2-135M-Instruct)
        VLLM_HOST: Host to bind (default: 0.0.0.0)
        VLLM_PORT: Port to use (default: 8001)
        VLLM_ENABLE_LORA: Enable LoRA support (default: true)
        VLLM_MAX_LORAS: Max concurrent LoRAs (default: 4)
        VLLM_MAX_LORA_RANK: Max LoRA rank (default: 64)
        VLLM_TENSOR_PARALLEL_SIZE: Tensor parallelism (default: 1)
        VLLM_GPU_MEMORY_UTIL: GPU memory utilization (default: 0.9)
        VLLM_DTYPE: Data type (default: auto)

    Returns:
        VLLMProcessManager if auto-start enabled, None otherwise
    """
    auto_start = os.environ.get("VLLM_AUTO_START", "false").lower() == "true"

    if not auto_start:
        return None

    print("Auto-starting vLLM server from environment configuration...")

    config = {
        "model": os.environ.get("VLLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        "host": os.environ.get("VLLM_HOST", "0.0.0.0"),
        "port": int(os.environ.get("VLLM_PORT", "8001")),
        "enable_lora": os.environ.get("VLLM_ENABLE_LORA", "true").lower() == "true",
        "max_loras": int(os.environ.get("VLLM_MAX_LORAS", "4")),
        "max_lora_rank": int(os.environ.get("VLLM_MAX_LORA_RANK", "64")),
        "tensor_parallel_size": int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.9")),
        "dtype": os.environ.get("VLLM_DTYPE", "auto"),
    }

    manager = VLLMProcessManager(**config)

    # Only start on rank 0 in distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        success = manager.start()
        if not success:
            print("WARNING: Failed to start vLLM server")
            return None
    else:
        print(f"Rank {local_rank}: Skipping vLLM server start (only rank 0 starts it)")

    return manager


# Global instance
_vllm_manager: Optional[VLLMProcessManager] = None


def get_manager() -> Optional[VLLMProcessManager]:
    """Get the global vLLM process manager."""
    return _vllm_manager


def initialize() -> Optional[VLLMProcessManager]:
    """Initialize vLLM process manager from environment."""
    global _vllm_manager

    if _vllm_manager is not None:
        return _vllm_manager

    _vllm_manager = create_from_env()
    return _vllm_manager


if __name__ == "__main__":
    # Test the process manager
    print("Testing vLLM Process Manager\n")

    manager = VLLMProcessManager(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        port=8001,
        enable_lora=True
    )

    try:
        if manager.start():
            print("\nServer started successfully!")
            print(f"Status: {manager.get_status()}")

            # Keep running
            print("\nPress Ctrl+C to stop...")
            while True:
                time.sleep(1)
        else:
            print("\nFailed to start server")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        manager.stop()
