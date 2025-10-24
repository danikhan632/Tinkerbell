import os
import io
import json

# Lazy initialization to avoid connection errors on startup
redis_client = None
minio_client = None
bucket_name = None

def _ensure_redis():
    """Lazy initialization of Redis client"""
    global redis_client
    if redis_client is None:
        try:
            import redis
            _redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            redis_client = redis.from_url(_redis_url)
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            redis_client = False  # Mark as failed
    return redis_client if redis_client is not False else None

def _ensure_minio():
    """Lazy initialization of MinIO client"""
    global minio_client, bucket_name
    if minio_client is None:
        try:
            from minio import Minio
            _minio_endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
            _minio_secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
            minio_client = Minio(
                _minio_endpoint,
                access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
                secure=_minio_secure,
            )
            bucket_name = os.environ.get("MINIO_BUCKET", "lora-adapters")
            # Ensure bucket exists
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
        except Exception as e:
            print(f"Warning: Could not connect to MinIO: {e}")
            minio_client = False  # Mark as failed
    return minio_client if minio_client is not False else None

def save_lora_metadata(adapter_id: str, meta: dict) -> None:
    client = _ensure_redis()
    if client:
        client.hset("lora_meta", adapter_id, json.dumps(meta))

def get_lora_metadata(adapter_id: str) -> dict | None:
    client = _ensure_redis()
    if not client:
        return None
    val = client.hget("lora_meta", adapter_id)
    if val is None:
        return None
    return json.loads(val)

def delete_lora_metadata(adapter_id: str) -> None:
    client = _ensure_redis()
    if client:
        client.hdel("lora_meta", adapter_id)

def upload_lora_weights(adapter_id: str, data: bytes) -> str:
    client = _ensure_minio()
    if not client:
        raise RuntimeError("MinIO not available")
    object_name = f"{adapter_id}.bin"
    client.put_object(bucket_name, object_name, io.BytesIO(data), length=len(data))
    return object_name

def delete_lora_weights(adapter_id: str) -> None:
    client = _ensure_minio()
    if not client:
        return
    object_name = f"{adapter_id}.bin"
    client.remove_object(bucket_name, object_name)