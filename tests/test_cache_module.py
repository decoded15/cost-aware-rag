from app.cache import get_cache_client

cache = get_cache_client()

try:
    if cache.ping():
        print("Cache connection successful!")
finally:
    cache.close()