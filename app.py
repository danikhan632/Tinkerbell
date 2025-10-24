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

# Flask application setup (no Celery - using worker thread instead)
app = Flask(__name__)

# In-memory stores (for prototyping)
futures_store: dict[str, dict] = {}
lora_adapters: dict[str, dict[str, dict]] = {}

# Thread safety for futures_store
futures_store_lock = threading.RLock()

# Share futures_store with worker module
import worker
worker.futures_store = futures_store
worker._futures_store_lock = futures_store_lock

# Supported models (example)
SUPPORTED_MODELS = [
    {"model_id": "llama-3-8b", "model_name": "meta-llama/Meta-Llama-3-8B", "arch": "llama"},
]

def generate_future_id() -> str:
    return f"future_{uuid.uuid4().hex[:8]}"

@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.error(traceback.format_exc())
    return jsonify({"detail": str(error)}), 500

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

@app.route("/fwd", methods=["POST"])
def fwd():
    future_id = generate_future_id()
    futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("forward", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/fwdbwd", methods=["POST"])
def fwdbwd():
    future_id = generate_future_id()
    futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("fwdbwd", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/optim_step", methods=["POST"])
def optim_step():
    future_id = generate_future_id()
    futures_store[future_id] = {"request": request.json, "status": "pending", "created_at": datetime.now(timezone.utc)}
    enqueue_job("optim", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

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

@app.route("/add_lora", methods=["POST"])
def add_lora():
    future_id = generate_future_id()
    enqueue_job("add_lora", future_id, request.json)
    model_id = request.json.get("base_model", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/remove_lora", methods=["POST"])
def remove_lora():
    future_id = generate_future_id()
    enqueue_job("remove_lora", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/load_weights", methods=["POST"])
def load_weights():
    future_id = generate_future_id()
    enqueue_job("load_weights", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/save_weights", methods=["POST"])
def save_weights():
    future_id = generate_future_id()
    enqueue_job("save_weights", future_id, request.json)
    model_id = request.json.get("model_id", "unknown")
    return jsonify({"request_id": future_id, "model_id": model_id})

@app.route("/save_weights_for_sampler", methods=["POST"])
def save_weights_for_sampler():
    future_id = generate_future_id()
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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)