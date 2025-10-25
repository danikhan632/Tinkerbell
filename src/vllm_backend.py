"""
vLLM backend for high-performance inference with LoRA adapter support.

This module provides in-process vLLM engine integration for:
- Fast inference using vLLM engine (co-located with training)
- Direct access to model weights and LoRA adapters
- No network overhead - same process, same GPU memory
- Immediate access to trained LoRA adapters
"""

import os
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch

# Try importing vLLM engine directly
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"vLLM not available: {e}")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None


# Global state
vllm_engine: Optional[LLM] = None
active_lora_adapters: Dict[str, str] = {}  # Maps adapter_id -> LoRA path
_engine_lock = threading.RLock()
_initialized = False


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine."""
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.2  # Only 5% - leave most for training
    dtype: str = "auto"
    enable_lora: bool = True
    max_loras: int = 4
    max_lora_rank: int = 64
    max_model_len: Optional[int] = None
    trust_remote_code: bool = True


def is_vllm_available() -> bool:
    """Check if vLLM is available."""
    return VLLM_AVAILABLE


def initialize_vllm_engine(config: Optional[VLLMConfig] = None) -> LLM:
    """
    Initialize the vLLM engine in-process (thread-safe).

    Args:
        config: vLLM configuration. If None, uses environment variables.

    Returns:
        vLLM LLM engine instance
    """
    global vllm_engine, _initialized

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is not available. Install with: pip install vllm")

    # Don't initialize vLLM in spawned child processes
    import multiprocessing
    if multiprocessing.current_process().name != 'MainProcess':
        print(f"Skipping vLLM initialization in child process: {multiprocessing.current_process().name}")
        return None

    with _engine_lock:
        # Double-check pattern to avoid reloading
        if _initialized and vllm_engine is not None:
            return vllm_engine

        if config is None:
            config = VLLMConfig(
                model=os.environ.get("VLLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
                tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
                gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.2")),  # 5% default
                dtype=os.environ.get("VLLM_DTYPE", "auto"),
                enable_lora=os.environ.get("VLLM_ENABLE_LORA", "true").lower() == "true",
                max_loras=int(os.environ.get("VLLM_MAX_LORAS", "4")),
                max_lora_rank=int(os.environ.get("VLLM_MAX_LORA_RANK", "64")),
                max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN")) if os.environ.get("VLLM_MAX_MODEL_LEN") else None,
                trust_remote_code=os.environ.get("VLLM_TRUST_REMOTE_CODE", "true").lower() == "true"
            )

        print(f"Initializing vLLM engine with model: {config.model}")
        print(f"  Tensor parallel size: {config.tensor_parallel_size}")
        print(f"  GPU memory utilization: {config.gpu_memory_utilization}")
        print(f"  LoRA support: {config.enable_lora}")
        if config.enable_lora:
            print(f"  Max LoRAs: {config.max_loras}, Max rank: {config.max_lora_rank}")

        # Initialize vLLM engine
        vllm_engine = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            enable_lora=config.enable_lora,
            max_loras=config.max_loras if config.enable_lora else None,
            max_lora_rank=config.max_lora_rank if config.enable_lora else None,
            max_model_len=config.max_model_len,
            trust_remote_code=config.trust_remote_code
        )

        _initialized = True
        print("vLLM engine initialized successfully (in-process, co-located)")
        return vllm_engine


def generate_with_vllm(
    prompts: List[str],
    lora_adapter_id: Optional[str] = None,
    n: int = 1,
    max_tokens: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    **generation_kwargs
) -> Dict[str, Any]:
    """
    Generate completions using vLLM engine with optional LoRA adapter.

    Args:
        prompts: List of text prompts
        lora_adapter_id: Optional LoRA adapter ID to use
        n: Number of completions per prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        min_p: Minimum probability threshold
        repetition_penalty: Repetition penalty
        **generation_kwargs: Additional parameters

    Returns:
        Dict with:
            - prompt_ids: List of prompt token IDs
            - completion_ids: List of completion token IDs
            - logprobs: List of log probabilities
    """
    global vllm_engine

    if vllm_engine is None:
        vllm_engine = initialize_vllm_engine()

    # Create sampling parameters
    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        logprobs=1,  # Return logprobs
        **generation_kwargs
    )

    # Prepare LoRA request if adapter specified
    lora_request = None
    if lora_adapter_id and lora_adapter_id in active_lora_adapters:
        lora_path = active_lora_adapters[lora_adapter_id]
        # Generate a consistent ID from the adapter name
        lora_int_id = abs(hash(lora_adapter_id)) % 10000
        lora_request = LoRARequest(
            lora_name=lora_adapter_id,
            lora_int_id=lora_int_id,
            lora_local_path=lora_path
        )
        print(f"Using LoRA adapter: {lora_adapter_id} from {lora_path}")

    # Generate with vLLM engine
    with _engine_lock:
        outputs = vllm_engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_request
        )

    # Convert vLLM outputs to expected format
    prompt_ids = []
    completion_ids = []
    completions_text = []
    logprobs = []

    for output in outputs:
        prompt_ids.append(output.prompt_token_ids)

        for completion in output.outputs:
            completion_ids.append(completion.token_ids)
            completions_text.append(completion.text)  # Add decoded text

            # Extract logprobs if available
            if completion.logprobs:
                comp_logprobs = [lp[tid].logprob for lp, tid in zip(completion.logprobs, completion.token_ids)]
                logprobs.append(comp_logprobs)
            else:
                logprobs.append(None)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "completions": completions_text,  # Add text completions
        "logprobs": logprobs,
        "prompt_logprobs": None  # vLLM doesn't return prompt logprobs by default
    }


def register_lora_adapter(adapter_id: str, lora_path: str) -> None:
    """
    Register a LoRA adapter for use with vLLM.

    Args:
        adapter_id: Unique identifier for the adapter
        lora_path: Path to LoRA weights directory (PEFT format)
    """
    with _engine_lock:
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA path does not exist: {lora_path}")

        active_lora_adapters[adapter_id] = lora_path
        print(f"Registered LoRA adapter '{adapter_id}' at {lora_path}")


def unregister_lora_adapter(adapter_id: str) -> bool:
    """
    Unregister a LoRA adapter.

    Args:
        adapter_id: Adapter ID to remove

    Returns:
        True if adapter was removed, False if not found
    """
    with _engine_lock:
        if adapter_id in active_lora_adapters:
            del active_lora_adapters[adapter_id]
            print(f"Unregistered LoRA adapter '{adapter_id}'")
            return True
        return False


def list_lora_adapters() -> List[Dict[str, str]]:
    """
    List all registered LoRA adapters.

    Returns:
        List of dicts with adapter_id and lora_path
    """
    with _engine_lock:
        return [
            {"adapter_id": aid, "lora_path": path}
            for aid, path in active_lora_adapters.items()
        ]


def get_engine_info() -> Dict[str, Any]:
    """
    Get information about the vLLM engine.

    Returns:
        Dict with engine configuration and status
    """
    global vllm_engine, _initialized

    if not _initialized or vllm_engine is None:
        return {
            "initialized": False,
            "available": VLLM_AVAILABLE,
            "active_loras": 0,
            "mode": "not_initialized"
        }

    with _engine_lock:
        return {
            "initialized": True,
            "available": VLLM_AVAILABLE,
            "active_loras": len(active_lora_adapters),
            "lora_adapters": list(active_lora_adapters.keys()),
            "mode": "in-process (co-located)"
        }


def save_and_register_lora(adapter_id: str, peft_model, base_path: str = "/tmp/lora_adapters") -> str:
    """
    Save a PEFT model and register it with vLLM.

    This bridges the gap between HuggingFace/Megatron training and vLLM sampling.

    Args:
        adapter_id: Unique adapter ID
        peft_model: PEFT model instance (from HuggingFace backend)
        base_path: Base directory to save adapters

    Returns:
        Path where adapter was saved
    """
    adapter_path = os.path.join(base_path, adapter_id)
    os.makedirs(adapter_path, exist_ok=True)

    print(f"Saving LoRA adapter '{adapter_id}' to {adapter_path}...")
    peft_model.save_pretrained(adapter_path)

    # Register with vLLM
    register_lora_adapter(adapter_id, adapter_path)

    return adapter_path
