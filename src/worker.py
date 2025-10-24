import os
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

# Try importing Megatron - it's optional
try:
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
    from megatron.training.tokenizer.tokenizer import _NullTokenizer
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.training import get_tokenizer
    MEGATRON_AVAILABLE = True
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


_work_queue: queue.Queue[tuple[str, str, dict]] = queue.Queue()

# Backend selection
USE_MEGATRON = MEGATRON_AVAILABLE and os.environ.get("USE_MEGATRON", "false").lower() == "true"

# Thread pool for concurrent job processing
# Each worker can process different adapters concurrently
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
_thread_pool: ThreadPoolExecutor = None
_futures_store_lock = threading.RLock()  # Lock for futures_store updates

def get_backend() -> str:
    """Get the current backend being used."""
    return "megatron" if USE_MEGATRON else "huggingface"

def _call_megatron(request_json: dict) -> dict:
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
    rank: int = int(os.environ.get("LOCAL_RANK", "0"))
    world_size: int = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

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
            if USE_MEGATRON:
                result = _call_megatron(params_json)
                if types and hasattr(types, 'SampleResponse'):
                    result = types.SampleResponse.model_validate(result)
            else:
                # HuggingFace generate - use simple dict access
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
            hf_backend.create_lora_adapter(model_id, lora_config)
            result = {"model_id": model_id}
        elif job_type == "remove_lora":
            # Make sure futures_store entry exists
            with _futures_store_lock:
                if request_id not in futures_store:
                    futures_store[request_id] = {"request": params_json, "status": "pending"}

            adapter_id = params_json.get("model_id") or params_json.get("adapter_id")
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

    # Initialize backend based on USE_MEGATRON flag
    if USE_MEGATRON:
        print("Initializing Megatron backend...")
        print("WARNING: Megatron backend does not yet support concurrent processing")
        # For Megatron, we still use sequential processing
        # Future work: support multiple Megatron model instances
        model, optimizer, data_iterator = _initialize_megatron()
    else:
        print("Initializing HuggingFace backend...")
        print(f"Worker pool size: {MAX_WORKERS} threads")
        hf_backend.initialize_base_model()

    # Create thread pool for concurrent job processing
    _thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="worker")

    print(f"Worker pool initialized. Concurrent job processing: {'Disabled (Megatron)' if USE_MEGATRON else 'Enabled (HF)'}")

    while True:
        job_type, request_id, params_json = _work_queue.get()

        if USE_MEGATRON:
            # Megatron: process sequentially (single-threaded)
            _process_megatron_job(job_type, request_id, params_json, model, optimizer, data_iterator)
        else:
            # HuggingFace: submit to thread pool for concurrent processing
            _thread_pool.submit(_process_job, job_type, request_id, params_json)


def _process_megatron_job(job_type: str, request_id: str, params_json: dict, model, optimizer, data_iterator):
    """Process Megatron job sequentially (not thread-safe)."""
    try:
        # Dispatch by job type
        if job_type in ("sample", "asample"):
            result = _call_megatron(params_json)
            if types and hasattr(types, 'SampleResponse'):
                result = types.SampleResponse.model_validate(result)
        elif job_type == "forward":
            result = _call_megatron_training("forward", params_json, model, optimizer, data_iterator)
        elif job_type == "fwdbwd":
            result = _call_megatron_training("fwdbwd", params_json, model, optimizer, data_iterator)
        elif job_type == "optim":
            result = _call_megatron_training("optim_step", params_json, model, optimizer, data_iterator)
        elif job_type == "add_lora":
            # Persist LoRA metadata and placeholder weights storage
            base_model = params_json.get('base_model', 'base')
            adapter_id = f"{base_model}_lora_{os.urandom(4).hex()}"
            meta = {key: params_json.get(key) for key in ('rank','alpha') if key in params_json}
            meta['created_at'] = datetime.now().isoformat()
            save_lora_metadata(adapter_id, meta)
            upload_lora_weights(adapter_id, b"")
            result = {"model_id": adapter_id}
        elif job_type == "remove_lora":
            adapter_id = params_json.get("model_id") or params_json.get("adapter_id")
            delete_lora_metadata(adapter_id)
            delete_lora_weights(adapter_id)
            result = {"status": "deleted"}
        elif job_type == "load_weights":
            result = {"status": "loaded"}
        elif job_type == "save_weights":
            result = {"status": "saved"}
        elif job_type == "save_weights_for_sampler":
            result = {"status": "saved"}
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        # Update futures_store
        if request_id in futures_store:
            futures_store[request_id]["result"] = result
            futures_store[request_id]["status"] = "completed"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if request_id not in futures_store:
            futures_store[request_id] = {"request": params_json, "status": "error"}
        futures_store[request_id]["status"] = "error"
        futures_store[request_id]["result"] = str(exc)

# Start worker thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()

def enqueue_job(job_type: str, request_id: str, params_json: dict) -> None:
    _work_queue.put((job_type, request_id, params_json))