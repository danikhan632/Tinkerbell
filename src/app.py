"""
Boilerplate Flask-based Tinker API Server

Run this script to start a minimal Tinker-compatible HTTP server with asynchronous job support via Celery.
"""
import os
import uuid
import traceback
from datetime import datetime, timezone

from flask import Flask, request, jsonify, abort

try:
    import tinker.types as types
except ImportError:
    print("Warning: tinker SDK not installed. Some features may not work.")
    types = None

from worker import enqueue_job
import threading

# Import vLLM process manager
import vllm_process_manager

# Flask application setup (no Celery - using worker thread instead)
app = Flask(__name__)

# vLLM process manager (will be initialized if VLLM_AUTO_START=true)
vllm_manager = None

# In-memory stores (for prototyping)
futures_store: dict[str, dict] = {}
lora_adapters: dict[str, dict[str, dict]] = {}

# Thread safety for futures_store
futures_store_lock = threading.RLock()

# Share futures_store with worker module
import worker
worker.futures_store = futures_store
worker._futures_store_lock = futures_store_lock

# Security middleware to reject proxy attempts
@app.before_request
def reject_proxy_requests():
    """Reject requests that try to use this server as a proxy."""
    # Check for absolute URLs (proxy attempts)
    if request.url_rule is None:  # No route matched
        path = request.path
        # Reject CONNECT method (used for proxying)
        if request.method == "CONNECT":
            return jsonify({"detail": "Proxy requests not allowed"}), 403
        # Reject if path contains external URLs
        if "://" in path or path.startswith("http"):
            return jsonify({"detail": "External URL requests not allowed"}), 403

# Supported models (example)
SUPPORTED_MODELS = [
    {"model_id": "llama-3-8b", "model_name": "meta-llama/Meta-Llama-3-8B", "arch": "llama"},
]

def generate_future_id() -> str:
    return f"future_{uuid.uuid4().hex[:8]}"

@app.errorhandler(404)
def handle_404(error):
    try:
        path = request.path if request else "unknown"
        method = request.method if request else "unknown"
        app.logger.warning(f"404 Not Found: {method} {path}")
    except:
        app.logger.warning("404 Not Found: Could not get request details")
    return jsonify({"detail": "Not Found"}), 404

@app.errorhandler(Exception)
def handle_exception(error):
    try:
        path = request.path if request else "unknown"
        method = request.method if request else "unknown"
        app.logger.error(f"Error on {method} {path}: {str(error)}")
        app.logger.error(traceback.format_exc())
    except:
        app.logger.error(f"Error: {str(error)}")
        app.logger.error(traceback.format_exc())
    return jsonify({"detail": str(error)}), 500

@app.route("/")
def index():
    """Root endpoint showing API info."""
    return jsonify({
        "name": "Tinkerbell API Server (Megatron-Bridge + vLLM)",
        "version": "2.0",
        "documentation": "https://tinker-docs.thinkingmachines.ai/async",
        "endpoints": {
            "health": "/healthz",
            "capabilities": "/get_server_capabilities",
            "futures": "/retrieve_future",
            "training": {
                "sync": {
                    "forward": "/api/v1/forward",
                    "forward_backward": "/api/v1/forward_backward",
                    "optim_step": "/api/v1/optim_step"
                },
                "async": {
                    "forward": "/api/v1/forward_async",
                    "forward_backward": "/api/v1/forward_backward_async",
                    "optim_step": "/api/v1/optim_step_async"
                }
            },
            "sampling": {
                "sync": "/api/v1/sample",
                "async": "/api/v1/asample"
            },
            "lora": {
                "sync": {
                    "add": "/api/v1/add_lora",
                    "remove": "/api/v1/remove_lora"
                },
                "async": {
                    "add": "/api/v1/add_lora_async",
                    "remove": "/api/v1/remove_lora_async"
                }
            },
            "weights": {
                "async": {
                    "load": "/api/v1/load_weights_async",
                    "save": "/api/v1/save_weights_async",
                    "save_for_sampler": "/api/v1/save_weights_for_sampler_async"
                }
            },
            "legacy": {
                "note": "Old endpoints (/fwd, /fwdbwd, etc.) still supported for backwards compatibility"
            }
        },
        "pattern": {
            "async": "Returns {request_id, model_id} immediately. Use /retrieve_future to get results.",
            "sync": "Waits for completion and returns results directly."
        }
    })

@app.route("/favicon.ico")
def favicon():
    # Return 204 No Content for favicon requests
    return "", 204

@app.route("/healthz", methods=["GET"])
def healthz():
    if types and hasattr(types, 'HealthResponse'):
        return jsonify(types.HealthResponse(status="ok").model_dump())
    return jsonify({"status": "ok"})

@app.route("/get_server_capabilities", methods=["GET"])
def get_server_capabilities():
    if types and hasattr(types, 'SupportedModel') and hasattr(types, 'GetServerCapabilitiesResponse'):
        models = [types.SupportedModel(**m).model_dump() for m in SUPPORTED_MODELS]
        return jsonify(types.GetServerCapabilitiesResponse(supported_models=models).model_dump())
    return jsonify({"supported_models": SUPPORTED_MODELS})

@app.route("/api/v1/asample", methods=["POST"])
def asample():
    params = types.SamplingAsampleParams.model_validate(request.json) if types and hasattr(types, 'SamplingAsampleParams') else None
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("asample", future_id, request.json)
    model_id = params.base_model if params and hasattr(params, 'base_model') else request.json.get("base_model", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/sample", methods=["POST"])
def sample():
    # Synchronous sampling - queue and wait for result
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("sample", future_id, request.json)

    # Wait for completion
    import time
    timeout = 60
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.5)
        elapsed += 0.5

    with futures_store_lock:
        if futures_store[future_id]["status"] == "error":
            error_msg = futures_store[future_id].get("result", "Unknown error")
            abort(500, f"Sampling failed: {error_msg}")
        result = futures_store[future_id].get("result")
    return jsonify(result if isinstance(result, dict) else {"result": str(result)})

# ============================================================================
# ASYNC ENDPOINTS (Return futures immediately - Tinker standard pattern)
# ============================================================================

@app.route("/api/v1/forward_async", methods=["POST"])
@app.route("/fwd", methods=["POST"])  # Legacy support
def forward_async():
    """Async forward pass - returns future immediately."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("forward", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/forward_backward_async", methods=["POST"])
@app.route("/fwdbwd", methods=["POST"])  # Legacy support
def forward_backward_async():
    """Async forward-backward pass - returns future immediately."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("fwdbwd", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/optim_step_async", methods=["POST"])
@app.route("/optim_step", methods=["POST"])  # Legacy support
def optim_step_async():
    """Async optimizer step - returns future immediately."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("optim", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

# ============================================================================
# SYNC ENDPOINTS (Wait for results before returning - Tinker standard pattern)
# ============================================================================

@app.route("/api/v1/forward", methods=["POST"])
def forward():
    """Sync forward pass - waits for result."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("forward", future_id, request.json)

    # Wait for completion
    import time
    timeout = 60
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.5)
        elapsed += 0.5

    with futures_store_lock:
        if futures_store[future_id]["status"] == "error":
            error_msg = futures_store[future_id].get("result", "Unknown error")
            abort(500, f"Forward failed: {error_msg}")
        result = futures_store[future_id].get("result")
    return jsonify(result if isinstance(result, dict) else {"result": str(result)})

@app.route("/api/v1/forward_backward", methods=["POST"])
def forward_backward():
    """Sync forward-backward pass - waits for result."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("fwdbwd", future_id, request.json)

    # Wait for completion
    import time
    timeout = 120  # Longer timeout for training
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.5)
        elapsed += 0.5

    with futures_store_lock:
        if futures_store[future_id]["status"] == "error":
            error_msg = futures_store[future_id].get("result", "Unknown error")
            abort(500, f"Forward-backward failed: {error_msg}")
        result = futures_store[future_id].get("result")
    return jsonify(result if isinstance(result, dict) else {"result": str(result)})

@app.route("/api/v1/optim_step", methods=["POST"])
def optim_step():
    """Sync optimizer step - waits for result."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("optim", future_id, request.json)

    # Wait for completion
    import time
    timeout = 60
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.5)
        elapsed += 0.5

    with futures_store_lock:
        if futures_store[future_id]["status"] == "error":
            error_msg = futures_store[future_id].get("result", "Unknown error")
            abort(500, f"Optimizer step failed: {error_msg}")
        result = futures_store[future_id].get("result")
    return jsonify(result if isinstance(result, dict) else {"result": str(result)})

@app.route("/retrieve_future", methods=["POST"])
def retrieve_future():
    # Simple validation without requiring FutureRetrieveParams
    future_id = request.json.get("request_id")
    if not future_id:
        abort(400, "request_id is required")

    with futures_store_lock:
        if future_id not in futures_store:
            abort(404, f"Future {future_id} not found")

        future = futures_store[future_id]
        status = future.get("status", "pending")

        # Check if job errored
        if status == "error":
            error_msg = future.get("result", "Unknown error")
            abort(500, f"Job failed: {error_msg}")

        # Check if job is still pending - return 202 without abort
        if status == "pending":
            return jsonify({"status": "pending"}), 202

        # Job completed successfully
        result = future.get("result")
        if result is None:
            return jsonify({"status": "completed", "result": None})

        return jsonify(result.model_dump() if hasattr(result, "model_dump") else result)

# ============================================================================
# LORA MANAGEMENT ENDPOINTS (Async by default)
# ============================================================================

@app.route("/api/v1/add_lora_async", methods=["POST"])
@app.route("/add_lora", methods=["POST"])  # Legacy support
def add_lora_async():
    """Add LoRA adapter - async (returns future immediately)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("add_lora", future_id, request.json)
    model_id = request.json.get("base_model", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/add_lora", methods=["POST"])
def add_lora():
    """Add LoRA adapter - sync (waits for result)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("add_lora", future_id, request.json)

    # Wait for completion
    import time
    timeout = 30
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.5)
        elapsed += 0.5

    with futures_store_lock:
        final_status = futures_store[future_id]["status"]
        if final_status == "error":
            error_msg = futures_store[future_id].get("result", "Unknown error")
            abort(500, f"Add LoRA failed: {error_msg}")
        elif final_status == "pending":
            app.logger.warning(f"Job {future_id} timed out after {timeout}s")
            abort(500, f"Add LoRA timed out after {timeout}s")
        result = futures_store[future_id].get("result")
        if result is None:
            app.logger.error(f"Job {future_id} completed but result is None. Status: {final_status}, Store: {futures_store[future_id]}")
    return jsonify(result if isinstance(result, dict) else {"result": str(result)})

@app.route("/api/v1/remove_lora_async", methods=["POST"])
@app.route("/remove_lora", methods=["POST"])  # Legacy support
def remove_lora_async():
    """Remove LoRA adapter - async (returns future immediately)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("remove_lora", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/remove_lora", methods=["POST"])
def remove_lora():
    """Remove LoRA adapter - sync (waits for result)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("remove_lora", future_id, request.json)

    # Wait for completion
    import time
    timeout = 30
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.5)
        elapsed += 0.5

    with futures_store_lock:
        if futures_store[future_id]["status"] == "error":
            error_msg = futures_store[future_id].get("result", "Unknown error")
            abort(500, f"Remove LoRA failed: {error_msg}")
        result = futures_store[future_id].get("result")
    return jsonify(result if isinstance(result, dict) else {"result": str(result)})

@app.route("/api/v1/load_weights_async", methods=["POST"])
@app.route("/load_weights", methods=["POST"])  # Legacy support
def load_weights_async():
    """Load weights - async (returns future immediately)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("load_weights", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/save_weights_async", methods=["POST"])
@app.route("/save_weights", methods=["POST"])  # Legacy support
def save_weights_async():
    """Save weights - async (returns future immediately)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("save_weights", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/api/v1/save_weights_for_sampler_async", methods=["POST"])
@app.route("/save_weights_for_sampler", methods=["POST"])  # Legacy support
def save_weights_for_sampler_async():
    """Save weights for sampler - async (returns future immediately)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("save_weights_for_sampler", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/get_info", methods=["POST"])
def get_info():
    model_id = request.json.get("model_id", "unknown")
    # TODO: implement model/adapter info lookup
    info = {
        "model_id": model_id,
        "model_data": {
            "model_name": model_id,
            "arch": "unknown"
        }
    }
    return jsonify(info)

@app.route("/register_custom_loss", methods=["POST"])
def register_custom_loss():
    """Register a custom loss function sent from the client."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("register_custom_loss", future_id, request.json)
    loss_name = request.json.get("loss_name", "unknown")
    return jsonify({"request_id": future_id, "loss_name": loss_name})

@app.route("/list_loss_functions", methods=["GET"])
def list_loss_functions():
    """List all available loss functions (built-in + custom)."""
    future_id = generate_future_id()
    with futures_store_lock:
        futures_store[future_id] = {"request": {}, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("list_loss_functions", future_id, {})

    # Wait for result (synchronous)
    import time
    timeout = 5
    elapsed = 0
    while True:
        with futures_store_lock:
            status = futures_store[future_id]["status"]
        if status != "pending" or elapsed >= timeout:
            break
        time.sleep(0.1)
        elapsed += 0.1

    with futures_store_lock:
        result = futures_store[future_id].get("result", {"available_loss_functions": []})
    return jsonify(result)

@app.route("/test_fwdbwd", methods=["GET"])
def test_fwdbwd():
    """Test the fwdbwd endpoint."""
    test_data = {
        "model_id": "test-model",
        "data": [[
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "test response"}
        ]],
        "loss_fn": "cross_entropy"
    }
    future_id = generate_future_id()
    futures_store[future_id] = {"request": test_data, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("fwdbwd", future_id, test_data)

    # Wait for the result
    import time
    timeout = 30
    elapsed = 0
    while futures_store[future_id]["status"] == "pending" and elapsed < timeout:
        time.sleep(1)
        elapsed += 1

    result = futures_store[future_id].get("result")
    if result:
        return jsonify(result if isinstance(result, dict) else {"result": str(result)})
    else:
        return jsonify({"error": "Test timed out or failed"}), 500

if __name__ == "__main__":
    # Initialize vLLM process manager if auto-start is enabled
    vllm_manager = vllm_process_manager.initialize()

    if vllm_manager and vllm_manager.is_running():
        print(f"\nvLLM server running at: {vllm_manager.base_url}")
        print(f"vLLM status: {vllm_manager.get_status()}\n")

        # Update environment for worker to connect to vLLM
        if not os.environ.get("VLLM_BASE_URL"):
            os.environ["VLLM_BASE_URL"] = vllm_manager.base_url
        if not os.environ.get("USE_VLLM"):
            os.environ["USE_VLLM"] = "true"

    # Start worker thread (only in main process)
    worker.start_worker()

    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Flask server on port {port}...\n")
    app.run(host="0.0.0.0", port=port)