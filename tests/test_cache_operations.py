
from app.cache import (
    get_cache_client,
    get_cached_response,
    cache_response,
)

cache = get_cache_client()
test_key = "test:rag:cache:v1"

sample_response = {
    "answer": "Semantic caching reuses answers to similar questions.",
    "sources": [{"source": "sample.txt", "chunk_index": 2}],
    "metrics": {"generation_ms": 0},
}

try:
    # Start with a clean test key.
    cache.delete(test_key)

    # First lookup: cache miss.
    result = get_cached_response(cache, test_key)
    print("First lookup:", "HIT" if result else "MISS")

    # Save the response.
    cache_response(cache, test_key, sample_response)

    # Second lookup: cache hit.
    result = get_cached_response(cache, test_key)
    print("Second lookup:", "HIT" if result else "MISS")
    print("Retrieved answer:", result["answer"])

    # Verify the TTL.
    print("TTL:", cache.ttl(test_key), "seconds")

finally:
    cache.delete(test_key)
    cache.close()
