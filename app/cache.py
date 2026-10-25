import json
import redis
import hashlib

def get_cache_client():
    """Create a client for our local Valkey server."""
    return redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )

CACHE_TTL_SECONDS = 600  # 10 minutes

def get_cached_response(cache, key: str) -> dict | None:
    """Return a cached response, or None if it doesn't exist."""
    cached_value = cache.get(key)

    if cached_value is None:
        return None

    return json.loads(cached_value)


def cache_response(cache, key: str, response: dict) -> None:
    """Save a response with an expiration time."""
    cache.set(
        key,
        json.dumps(response),
        ex=CACHE_TTL_SECONDS,
    )

def make_cache_key(question: str) -> str:
    """Create a stable key for a normalized question."""
    normalized_question = " ".join(question.strip().lower().split())

    question_hash = hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()

    return f"rag:v1:{question_hash}"


