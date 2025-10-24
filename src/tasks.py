"""
Celery tasks for the Flask-based Tinker API Server.

Note: These tasks are registered with the celery app in app.py
"""

from datetime import datetime
try:
    import tinker.types as types
except ImportError:
    print("Warning: tinker SDK not installed. Some features may not work.")
    types = None

from worker import enqueue_job


def forward_task(request_id: str, params_json: dict) -> None:
    if types:
        types.TrainingForwardParams.model_validate(params_json)
    enqueue_job("forward", request_id, params_json)


def fwdbwd_task(request_id: str, params_json: dict) -> None:
    if types:
        types.TrainingForwardBackwardParams.model_validate(params_json)
    enqueue_job("fwdbwd", request_id, params_json)


def optim_step_task(request_id: str, params_json: dict) -> None:
    if types:
        types.TrainingOptimStepParams.model_validate(params_json)
    enqueue_job("optim", request_id, params_json)


def add_lora_task(request_id: str, params_json: dict) -> None:
    if types:
        types.LoraAddParams.model_validate(params_json)
    enqueue_job("add_lora", request_id, params_json)


def remove_lora_task(request_id: str, params_json: dict) -> None:
    if types:
        types.LoraRemoveParams.model_validate(params_json)
    enqueue_job("remove_lora", request_id, params_json)


def load_weights_task(request_id: str, params_json: dict) -> None:
    if types:
        types.WeightLoadParams.model_validate(params_json)
    enqueue_job("load_weights", request_id, params_json)


def save_weights_task(request_id: str, params_json: dict) -> None:
    if types:
        types.WeightSaveParams.model_validate(params_json)
    enqueue_job("save_weights", request_id, params_json)


def save_weights_for_sampler_task(request_id: str, params_json: dict) -> None:
    if types:
        types.WeightSaveForSamplerParams.model_validate(params_json)
    enqueue_job("save_weights_for_sampler", request_id, params_json)


def asample_task(request_id: str, params_json: dict) -> None:
    if types:
        types.SamplingAsampleParams.model_validate(params_json)
    enqueue_job("asample", request_id, params_json)


def sample_task(request_id: str, params_json: dict) -> None:
    if types:
        types.SamplingSampleParams.model_validate(params_json)
    enqueue_job("sample", request_id, params_json)