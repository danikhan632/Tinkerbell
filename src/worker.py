import os
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
import multiprocessing

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

# Try importing Megatron - it's optional
try:
    MEGATRON_AVAILABLE = True
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.datasets.utils import compile_helpers
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
    from megatron.bridge.training.tokenizers.tokenizer import _NullTokenizer
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

except ImportError:
    MEGATRON_AVAILABLE = False
    print("Megatron-LM not available. Using HuggingFace backend only.")

import httpx

try:
    import tinker.types as types
except ImportError:
    print("Warning: tinker SDK not installed. Some features may not work.")
    types = None

# Avoid circular imports - futures_store will be set by app.py
futures_store = {}

from storage import (
    save_lora_metadata,
    delete_lora_metadata,
    upload_lora_weights,
    delete_lora_weights,
)

# Import HuggingFace backend
import hf_backend

# Import Megatron backend
if MEGATRON_AVAILABLE:
    import megatron_backend
else:
    megatron_backend = None

# Import vLLM backend
try:
    import vllm_backend
    VLLM_BACKEND_AVAILABLE = vllm_backend.is_vllm_available()
except ImportError:
    print("vLLM backend not available. Install with: pip install vllm requests")
    vllm_backend = None
    VLLM_BACKEND_AVAILABLE = False


_work_queue: queue.Queue[tuple[str, str, dict]] = queue.Queue()

# Backend selection
USE_MEGATRON = MEGATRON_AVAILABLE and os.environ.get("USE_MEGATRON", "false").lower() == "true"
USE_VLLM = VLLM_BACKEND_AVAILABLE and os.environ.get("USE_VLLM", "false").lower() == "true"

# Thread pool for concurrent job processing
# Each worker can process different adapters concurrently
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
_thread_pool: ThreadPoolExecutor = None
_futures_store_lock = threading.RLock()  # Lock for futures_store updates

def get_backend() -> str:
    """Get the current backend being used."""
    if USE_MEGATRON:
        return "megatron"
    elif USE_VLLM:
        return "vllm"
    else:
        return "huggingface"

def _call_vllm(request_json: dict) -> dict:
    """Call vLLM backend for sampling with LoRA adapter support."""
    params = request_json.get("sampling_params", {})
    chunks = request_json.get("prompt", {}).get("chunks", [])

    # Extract prompts from chunks or direct prompts field
    try:
        from megatron.training import get_tokenizer
        tokenizer = get_tokenizer()
        prompts = [tokenizer.detokenize(chunks[0]["tokens"])] if chunks else []
    except:
        prompts = request_json.get("prompts", [])

    # Filter out empty prompts and validate
    prompts = [p for p in prompts if p and p.strip()]

    if not prompts:
        raise ValueError("No valid prompts provided. Prompts cannot be empty.")

    # Get LoRA adapter if specified
    lora_adapter_id = request_json.get("model_id") or request_json.get("adapter_id")

    # Call vLLM with parameters
    vllm_result = vllm_backend.generate_with_vllm(
        prompts=prompts,
        lora_adapter_id=lora_adapter_id,
        n=params.get("n", 1),
        max_tokens=params.get("max_tokens", 16),
        temperature=params.get("temperature", 1.0),
        top_p=params.get("top_p", 1.0),
        top_k=params.get("top_k", -1),
        min_p=params.get("min_p", 0.0),
        repetition_penalty=params.get("repetition_penalty", 1.0)
    )

    # Convert vLLM response to expected format
    prompt_ids = vllm_result.get("prompt_ids", [])
    completion_ids = vllm_result.get("completion_ids", [])
    completions_text = vllm_result.get("completions", [])
    logprobs = vllm_result.get("logprobs", [])

    # Build sequences in expected format
    sequences = []
    for i, comp_ids in enumerate(completion_ids):
        comp_logprobs = logprobs[i] if i < len(logprobs) else None
        max_expected = params.get("max_tokens", 16)
        stop_reason = "length" if len(comp_ids) >= max_expected else "stop"
        sequences.append({
            "stop_reason": stop_reason,
            "tokens": comp_ids,
            "logprobs": comp_logprobs
        })

    return {
        "sequences": sequences,
        "completions": completions_text,  # Add text completions
        "prompt_logprobs": vllm_result.get("prompt_logprobs")
    }

def _call_megatron(request_json: dict) -> dict:
    """Call Megatron server for sampling."""
    server_url = os.environ.get("MEGATRON_SERVER_URL", "http://localhost:5000")
    params = request_json.get("sampling_params", {})
    chunks = request_json.get("prompt", {}).get("chunks", [])
    try:
        from megatron.training import get_tokenizer
        tokenizer = get_tokenizer()
        prompts = [tokenizer.detokenize(chunks[0]["tokens"])] if chunks else []
    except:
        prompts = request_json.get("prompts", [])
    meg_req: dict = {"prompts": prompts}
    if params.get("max_tokens") is not None:
        meg_req["tokens_to_generate"] = params["max_tokens"]
    if params.get("temperature") is not None:
        meg_req["temperature"] = params["temperature"]
    if params.get("top_k") is not None:
        meg_req["top_k"] = params["top_k"]
    if params.get("top_p") is not None:
        meg_req["top_p"] = params["top_p"]
    resp = httpx.put(f"{server_url}/api", json=meg_req, timeout=None).json()
    prev_len = len(chunks[0]["tokens"]) if chunks else 0
    full_tokens = resp.get("tokens", [])[0]
    gen = full_tokens[prev_len:]
    full_logprobs = resp.get("logprobs")
    if full_logprobs is not None:
        full_logprobs = full_logprobs[0]
        prompt_logprobs = full_logprobs[:prev_len]
        gen_logprobs = full_logprobs[prev_len:]
    else:
        prompt_logprobs = None
        gen_logprobs = None
    stop = "length" if len(gen) == meg_req.get("tokens_to_generate", 0) else "stop"
    seq = {"stop_reason": stop, "tokens": gen, "logprobs": gen_logprobs}
    return {"sequences": [seq], "prompt_logprobs": prompt_logprobs}

def _initialize_megatron(tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1) -> tuple[GPTModel, Adam, object]:
    """
    Initialize torch.distributed and Megatron-Core model parallel groups.

    Args:
        tensor_model_parallel_size: Number of GPUs for tensor model parallelism.
        pipeline_model_parallel_size: Number of GPUs for pipeline model parallelism.
    """
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank: int = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size: int = int(os.environ.get("WORLD_SIZE", str(torch.cuda.device_count())))

    # For single-machine setup, set environment variables if not already set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    # Initialize process group with proper backend and init method
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",  # Use NCCL for NVIDIA GPUs
            init_method="env://",  # Use environment variables
            world_size=world_size,
            rank=rank
        )

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )

    model_parallel_cuda_manual_seed(123)

    transformer_config: TransformerConfig = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )

    gpt_model: GPTModel = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=64,
    )

    device: torch.device = torch.device("cuda")
    gpt_model.to(device)

    # Wrap model with DistributedDataParallel for proper gradient synchronization.
    # This provides the finish_grad_sync() method required by finalize_model_grads().
    config: TransformerConfig = gpt_model.config
    ddp_config: DistributedDataParallelConfig = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=gpt_model,
    )

    optim: Adam = Adam(gpt_model.parameters())

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    dataset_config: GPTDatasetConfig = GPTDatasetConfig(
        random_seed=0,
        sequence_length=64,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=64),
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, dataset_config
    ).build()

    train_dataloader: object = iter(DataLoader(datasets[0], batch_size=8, shuffle=True))

    return gpt_model, optim, train_dataloader

def _forward_step(data_iterator, model):
    """Forward step function."""
    data = next(data_iterator)
    tokens = data["tokens"].to(torch.cuda.current_device())
    attention_mask = data["attention_mask"].to(torch.cuda.current_device())
    position_ids = data["position_ids"].to(torch.cuda.current_device())
    labels = data["labels"].to(torch.cuda.current_device())

    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels
    )

    def loss_func(output_tensor):
        loss = output_tensor
        return loss, {"loss": loss}

    return output_tensor, loss_func

def _call_megatron_training(job_type: str, params: dict, model: GPTModel, optimizer: Adam, data_iterator: object) -> dict:
    if job_type == "forward":
        output_tensor, loss_func = _forward_step(data_iterator, model)
        loss = loss_func(output_tensor)
        return types.FwdBwdOutput(loss_fn_outputs=loss, metrics={})
    elif job_type == "fwdbwd":
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=_forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=1,
            seq_length=64,
            micro_batch_size=8,
            decoder_seq_length=64,
            forward_only=False,
        )
        return types.FwdBwdOutput(loss_fn_outputs=losses_reduced, metrics={})
    elif job_type == "optim_step":
        finalize_model_grads([model])
        grads_for_norm = []
        for param in model.parameters():
            if param.grad is not None:
                grads_for_norm.append(param.grad)
        grad_norm = get_grad_norm_fp32(grads_for_norm)
        old_params = [p.clone() for p in model.parameters()]
        optimizer.step()
        weight_norm = torch.cat([p.view(-1) for p in model.parameters()]).norm()
        update_norm = torch.cat([(p - old_p).view(-1) for p, old_p in zip(model.parameters(), old_params)]).norm()
        optimizer.zero_grad()
        return types.OptimStepResponse(grad_norm=grad_norm, weight_norm=weight_norm, update_norm=update_norm)
    else:
        raise ValueError(f"Unknown job type: {job_type}")

def _process_job(job_type: str, request_id: str, params_json: dict) -> None:
    """
    Process a single job. This runs in a thread pool worker.

    Args:
        job_type: Type of job to process
        request_id: Unique identifier for this job
        params_json: Job parameters
    """
    try:
        result = None

        # Dispatch by job type and backend
        if job_type in ("sample", "asample"):
            if USE_VLLM:
                # Use vLLM for high-performance sampling with LoRA support
                result = _call_vllm(params_json)
                if types and hasattr(types, 'SampleResponse'):
                    result = types.SampleResponse.model_validate(result)
            elif USE_MEGATRON:
                # Use Megatron server for sampling
                result = _call_megatron(params_json)
                if types and hasattr(types, 'SampleResponse'):
                    result = types.SampleResponse.model_validate(result)
            else:
                # Fallback to HuggingFace for sampling
                model_id = params_json.get("model_id", "base")
                base_model = params_json.get("base_model", "base")

                # Extract messages - try different formats
                messages = []

                # Try direct prompts list
                if "prompts" in params_json and params_json["prompts"]:
                    prompts = params_json["prompts"]
                    messages = [{"role": "user", "content": prompts[0]} if isinstance(prompts, list) else {"role": "user", "content": prompts}]
                # Try prompt chunks format
                elif "prompt" in params_json:
                    chunks = params_json.get("prompt", {}).get("chunks", [])
                    if chunks and len(chunks) > 0 and "tokens" in chunks[0]:
                        messages = [{"role": "user", "content": hf_backend.tokenizer.decode(chunks[0]["tokens"])}]

                # Default if no messages found
                if not messages:
                    messages = [{"role": "user", "content": "Hello"}]

                gen_params = params_json.get("sampling_params", {})
                result = hf_backend.generate(
                    model_id=base_model if base_model != "base" else model_id,
                    messages=messages,
                    max_new_tokens=gen_params.get("max_tokens", 50),
                    temperature=gen_params.get("temperature", 1.0),
                    top_p=gen_params.get("top_p", 1.0),
                    top_k=gen_params.get("top_k", 50)
                )
        elif job_type == "forward":
            model_id = params_json.get("model_id", "base")
            data = params_json.get("data", [])
            result = hf_backend.forward_backward(
                model_id=model_id,
                data=data,
                loss_fn="cross_entropy",
                loss_fn_inputs=None,
                lora_config=None
            )
        elif job_type == "fwdbwd":
            model_id = params_json.get("model_id", "base")
            data = params_json.get("data", [])
            loss_fn = params_json.get("loss_fn", "cross_entropy")
            loss_fn_inputs = params_json.get("loss_fn_inputs")
            result = hf_backend.forward_backward(
                model_id=model_id,
                data=data,
                loss_fn=loss_fn,
                loss_fn_inputs=loss_fn_inputs,
                lora_config=None
            )
        elif job_type == "optim":
            model_id = params_json.get("model_id", "base")
            adam_params = params_json.get("adam_params", {})
            result = hf_backend.optim_step(
                model_id=model_id,
                adam_params=hf_backend.AdamParams(**adam_params) if adam_params else None
            )
        elif job_type == "add_lora":
            # Make sure futures_store entry exists
            with _futures_store_lock:
                if request_id not in futures_store:
                    futures_store[request_id] = {"request": params_json, "status": "pending"}

            base_model = params_json.get('base_model', 'base')
            model_id = f"{base_model}_lora_{os.urandom(4).hex()}"
            lora_config = hf_backend.LoraConfigParams(
                r=params_json.get('rank', 16),
                lora_alpha=params_json.get('alpha', 32)
            )
            adapter = hf_backend.create_lora_adapter(model_id, lora_config)

            # Automatically save and register with vLLM for co-located sampling
            if USE_VLLM and VLLM_BACKEND_AVAILABLE:
                try:
                    adapter_path = vllm_backend.save_and_register_lora(model_id, adapter)
                    print(f"LoRA adapter '{model_id}' ready for vLLM sampling at {adapter_path}")
                except Exception as e:
                    print(f"Warning: Could not register LoRA with vLLM: {e}")

            result = {"model_id": model_id}
        elif job_type == "remove_lora":
            # Make sure futures_store entry exists
            with _futures_store_lock:
                if request_id not in futures_store:
                    futures_store[request_id] = {"request": params_json, "status": "pending"}

            adapter_id = params_json.get("model_id") or params_json.get("adapter_id")

            # Unregister from vLLM first
            if USE_VLLM and VLLM_BACKEND_AVAILABLE:
                vllm_backend.unregister_lora_adapter(adapter_id)

            # Delete from HuggingFace backend
            hf_backend.delete_adapter(adapter_id)
            result = {"status": "deleted"}
        elif job_type == "load_weights":
            result = {"status": "loaded"}
        elif job_type == "save_weights":
            result = {"status": "saved"}
        elif job_type == "save_weights_for_sampler":
            result = {"status": "saved"}
        elif job_type == "list_loss_functions":
            result = hf_backend.list_loss_functions()
        elif job_type == "register_custom_loss":
            # Deserialize and register custom loss function
            loss_name = params_json.get("loss_name")
            loss_fn_serialized = params_json.get("loss_fn_serialized")

            if not loss_name or not loss_fn_serialized:
                raise ValueError("register_custom_loss requires 'loss_name' and 'loss_fn_serialized'")

            # Deserialize the function using cloudpickle
            import cloudpickle
            import base64
            loss_fn_bytes = base64.b64decode(loss_fn_serialized)
            loss_fn = cloudpickle.loads(loss_fn_bytes)

            # Register with the loss function registry
            import loss_functions
            loss_functions.register_custom_loss(loss_name, loss_fn)

            result = {
                "status": "registered",
                "loss_name": loss_name,
                "message": f"Custom loss '{loss_name}' successfully registered"
            }
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        # Update futures_store with result (thread-safe)
        with _futures_store_lock:
            if request_id in futures_store:
                futures_store[request_id]["result"] = result
                futures_store[request_id]["status"] = "completed"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Update futures_store with error (thread-safe)
        with _futures_store_lock:
            if request_id not in futures_store:
                futures_store[request_id] = {"request": params_json, "status": "error"}
            futures_store[request_id]["status"] = "error"
            futures_store[request_id]["result"] = str(exc)


def _worker_loop() -> None:
    """
    Main worker loop that dispatches jobs to a thread pool.
    This allows concurrent processing of different adapters.
    """
    global _thread_pool

    print("[_worker_loop] Worker loop thread started!")
    print(f"[_worker_loop] USE_MEGATRON={USE_MEGATRON}, USE_VLLM={USE_VLLM}")

    try:
        # Initialize backend based on flags
        if USE_MEGATRON:
            print("Initializing Megatron backend...")
            # Get model name from environment
            model_name = os.environ.get("MEGATRON_MODEL") or os.environ.get("VLLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

            # Initialize Megatron backend with PEFT support
            if megatron_backend:
                # Initialize the bridge with the model name
                megatron_backend.initialize_base_model(
                    model_name_or_path=model_name,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    trust_remote_code=True,
                    load_weights=True
                )
                print("Megatron backend initialized with LoRA support (PEFT)")
                print("  Concurrent adapter creation: Enabled")
                print("  Training mode: Sequential (PEFT limitation)")
                print("[_worker_loop] Megatron-Bridge initialized, skipping Megatron-Core")

                # Don't initialize Megatron-Core distributed training
                # We're using Megatron-Bridge for weight management only
                model, optimizer, data_iterator = None, None, None
                print("[_worker_loop] Set model=None, continuing...")
            else:
                # Initialize Megatron model (for direct training if needed)
                model, optimizer, data_iterator = _initialize_megatron()
        else:
            print("[_worker_loop] Not using Megatron")
            model, optimizer, data_iterator = None, None, None
            # Initialize HuggingFace backend if not using Megatron and not using vLLM-only
            if not USE_VLLM:
                print("Initializing HuggingFace backend...")
                print(f"Worker pool size: {MAX_WORKERS} threads")
                hf_backend.initialize_base_model()
            else:
                print("Using vLLM-only mode (no HuggingFace backend)")
                print("WARNING: Training operations will not work in vLLM-only mode")

        print("[_worker_loop] Backend initialization complete, checking vLLM...")
        # Initialize vLLM engine after training backend
        if USE_VLLM:
            print("\n[_worker_loop] Initializing vLLM engine (in-process, co-located)...")

            # If HuggingFace is also loaded, warn about memory usage
            if not USE_MEGATRON and hf_backend.base_model is not None:
                print("WARNING: Both HuggingFace and vLLM are loading models!")
                print("This may cause OOM. Consider one of:")
                print("  1. Use vLLM only: export USE_VLLM=true (no training)")
                print("  2. Use HuggingFace only: export USE_VLLM=false")
                print("  3. Use Megatron + vLLM: export USE_MEGATRON=true USE_VLLM=true")

            print("[_worker_loop] Calling vllm_backend.initialize_vllm_engine()...")
            vllm_backend.initialize_vllm_engine()
            print("[_worker_loop] vLLM engine initialized successfully")
            print("vLLM engine initialized for high-performance sampling with LoRA support")
            print("  Mode: Co-located (same process, same GPU memory, immediate LoRA access)")
        else:
            print("[_worker_loop] Skipping vLLM initialization")

        print("[_worker_loop] Creating thread pool...")
        # Create thread pool for concurrent job processing
        _thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="worker")
        print("[_worker_loop] Thread pool created")

        backend_info = []
        if USE_VLLM:
            backend_info.append("vLLM (sampling)")
        if USE_MEGATRON:
            backend_info.append("Megatron (training)")
        if not USE_MEGATRON:
            backend_info.append("HuggingFace (training/sampling)")

        print(f"Worker pool initialized. Active backends: {', '.join(backend_info)}")
        if USE_MEGATRON:
            print(f"Concurrent job processing: Sequential training (PEFT multi-adapter)")
        else:
            print(f"Concurrent job processing: Enabled (HuggingFace)")

        print("[_worker_loop] Starting main worker loop...")
        while True:
            print(f"[_worker_loop] Waiting for job... (queue size: {_work_queue.qsize()})")
            job_type, request_id, params_json = _work_queue.get()
            print(f"[_worker_loop] Got job: type={job_type}, request_id={request_id}")

            if USE_MEGATRON:
                # Megatron: process sequentially (single-threaded)
                print(f"[_worker_loop] Processing with Megatron backend...")
                _process_megatron_job(job_type, request_id, params_json, model, optimizer, data_iterator)
            else:
                # HuggingFace: submit to thread pool for concurrent processing
                print(f"[_worker_loop] Submitting to thread pool...")
                _thread_pool.submit(_process_job, job_type, request_id, params_json)
    except Exception as e:
        print(f"[_worker_loop] FATAL ERROR in worker loop: {e}")
        import traceback
        traceback.print_exc()
        raise


def _process_megatron_job(job_type: str, request_id: str, params_json: dict, model, optimizer, data_iterator):
    """Process Megatron job sequentially (not thread-safe)."""
    try:
        # Dispatch by job type
        if job_type in ("sample", "asample"):
            if USE_VLLM:
                # Use vLLM for sampling instead of Megatron server
                result = _call_vllm(params_json)
            else:
                # Use Megatron server
                result = _call_megatron(params_json)
            if types and hasattr(types, 'SampleResponse'):
                result = types.SampleResponse.model_validate(result)
        elif job_type == "forward":
            result = _call_megatron_training("forward", params_json, model, optimizer, data_iterator)
        elif job_type == "fwdbwd":
            # Use megatron_backend for LoRA training
            model_id = params_json.get("model_id") or params_json.get("adapter_id", "default")
            data = params_json.get("data", [])
            loss_fn = params_json.get("loss_fn", "cross_entropy")
            loss_fn_inputs = params_json.get("loss_fn_inputs", {})

            # Create LoRA config from params
            lora_config = None
            if "rank" in params_json or "alpha" in params_json:
                lora_config = megatron_backend.LoraConfigParams(
                    r=params_json.get("rank", 16),
                    lora_alpha=params_json.get("alpha", 32),
                    lora_dropout=params_json.get("lora_dropout", 0.1),
                )

            result = megatron_backend.forward_backward(
                model_id=model_id,
                data=data,
                loss_fn=loss_fn,
                loss_fn_inputs=loss_fn_inputs,
                lora_config=lora_config
            )
        elif job_type == "optim":
            # Use megatron_backend for optimizer step
            model_id = params_json.get("model_id") or params_json.get("adapter_id", "default")
            adam_params = None
            if "learning_rate" in params_json:
                adam_params = megatron_backend.AdamParams(
                    learning_rate=params_json.get("learning_rate", 1e-4),
                    beta1=params_json.get("beta1", 0.9),
                    beta2=params_json.get("beta2", 0.999),
                    eps=params_json.get("eps", 1e-8),
                    weight_decay=params_json.get("weight_decay", 0.0),
                )

            result = megatron_backend.optim_step(model_id, adam_params)
        elif job_type == "add_lora":
            # Create LoRA adapter using megatron_backend
            base_model = params_json.get('base_model', 'base')
            adapter_id = f"{base_model}_lora_{os.urandom(4).hex()}"
            print(f"[add_lora] Creating adapter '{adapter_id}' with base_model='{base_model}'")

            # Create LoRA config from params
            lora_config = megatron_backend.LoraConfigParams(
                r=params_json.get("rank", 16),
                lora_alpha=params_json.get("alpha", 32),
                lora_dropout=params_json.get("lora_dropout", 0.1),
                target_modules=params_json.get("target_modules"),
            )
            print(f"[add_lora] LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")

            # Create adapter
            adapter_info = megatron_backend.create_lora_adapter(adapter_id, lora_config)
            print(f"[add_lora] Adapter created: {adapter_info}")

            # Persist metadata
            meta = {key: params_json.get(key) for key in ('rank','alpha') if key in params_json}
            meta['created_at'] = datetime.now().isoformat()
            save_lora_metadata(adapter_id, meta)
            print(f"[add_lora] Metadata saved")

            # Save adapter weights for vLLM if enabled
            if USE_VLLM and vllm_backend:
                try:
                    # Get the PEFT model from megatron_backend
                    peft_model = megatron_backend.lora_adapters.get(adapter_id)
                    if peft_model:
                        adapter_path = vllm_backend.save_and_register_lora(adapter_id, peft_model)
                        print(f"LoRA adapter '{adapter_id}' ready for vLLM sampling at {adapter_path}")
                    else:
                        print(f"[add_lora] Warning: No PEFT model found in megatron_backend.lora_adapters")
                except Exception as e:
                    print(f"Warning: Could not register LoRA with vLLM: {e}")

            result = {"model_id": adapter_id}
            print(f"[add_lora] Returning result: {result}")
        elif job_type == "remove_lora":
            # Remove adapter using megatron_backend
            adapter_id = params_json.get("model_id") or params_json.get("adapter_id")
            megatron_backend.remove_lora_adapter(adapter_id)
            delete_lora_metadata(adapter_id)
            delete_lora_weights(adapter_id)

            # Unregister from vLLM if enabled
            if USE_VLLM and vllm_backend:
                vllm_backend.unregister_lora_adapter(adapter_id)

            result = {"status": "deleted"}
        elif job_type == "load_weights":
            result = {"status": "loaded"}
        elif job_type == "save_weights":
            result = {"status": "saved"}
        elif job_type == "save_weights_for_sampler":
            result = {"status": "saved"}
        elif job_type == "list_loss_functions":
            if megatron_backend:
                result = megatron_backend.list_loss_functions()
            else:
                import loss_functions
                result = {
                    "available_loss_functions": loss_functions.LOSS_REGISTRY.list_available()
                }
        elif job_type == "register_custom_loss":
            # Deserialize and register custom loss function
            loss_name = params_json.get("loss_name")
            loss_fn_serialized = params_json.get("loss_fn_serialized")

            if not loss_name or not loss_fn_serialized:
                raise ValueError("register_custom_loss requires 'loss_name' and 'loss_fn_serialized'")

            # Deserialize the function using cloudpickle
            import cloudpickle
            import base64
            loss_fn_bytes = base64.b64decode(loss_fn_serialized)
            loss_fn = cloudpickle.loads(loss_fn_bytes)

            # Register with the loss function registry
            import loss_functions
            loss_functions.register_custom_loss(loss_name, loss_fn)

            result = {
                "status": "registered",
                "loss_name": loss_name,
                "message": f"Custom loss '{loss_name}' successfully registered"
            }
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        # Update futures_store (with lock for thread safety)
        with _futures_store_lock:
            if request_id in futures_store:
                futures_store[request_id]["result"] = result
                futures_store[request_id]["status"] = "completed"
                print(f"[worker] Job {request_id} completed. Result type: {type(result)}, Result: {result}")
            else:
                print(f"[worker] ERROR: Job {request_id} not found in futures_store!")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with _futures_store_lock:
            if request_id not in futures_store:
                futures_store[request_id] = {"request": params_json, "status": "error"}
            futures_store[request_id]["status"] = "error"
            futures_store[request_id]["result"] = str(exc)

# Worker thread - will be started by start_worker()
_worker_thread = None
_worker_started = False

def start_worker():
    """Start the worker thread. Call this from main process only."""
    global _worker_thread, _worker_started

    # Don't start worker in spawned child processes
    if multiprocessing.current_process().name != 'MainProcess':
        print(f"Skipping worker start in child process: {multiprocessing.current_process().name}")
        return

    if _worker_started:
        print("Worker already started")
        return

    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()
        _worker_started = True
        print("Worker thread started")

def enqueue_job(job_type: str, request_id: str, params_json: dict) -> None:
    print(f"[enqueue_job] Queuing job: type={job_type}, request_id={request_id}")
    _work_queue.put((job_type, request_id, params_json))
    print(f"[enqueue_job] Job queued. Queue size: {_work_queue.qsize()}")