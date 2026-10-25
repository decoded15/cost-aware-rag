
import uuid

from app.cache import (
    get_cache_client,
    get_cached_response,
    cache_response,
    make_cache_key,
)


def test_cache_miss_and_hit():
    cache = get_cache_client()
    key = f"test:rag:{uuid.uuid4().hex}"
    expected = {"answer": "Caching works!", "sources": []}

    try:
        assert get_cached_response(cache, key) is None

        cache_response(cache, key, expected)

        assert get_cached_response(cache, key) == expected
    finally:
        cache.delete(key)
        cache.close()


def test_cache_key_normalization():
    assert make_cache_key(
        "What is semantic caching?"
    ) == make_cache_key(
        "  WHAT IS SEMANTIC CACHING?  "
    )


def test_cache_key_distinguishes_questions():
    assert make_cache_key(
        "What is semantic caching?"
    ) != make_cache_key(
        "What is semantic caching"
    )


def test_cached_response_has_ttl():
    cache = get_cache_client()
    key = f"test:rag:{uuid.uuid4().hex}"

    try:
        cache_response(cache, key, {"answer": "Test"})
        ttl = cache.ttl(key)

        assert 0 < ttl <= 600
    finally:
        cache.delete(key)
        cache.close()
