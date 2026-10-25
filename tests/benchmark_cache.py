
from statistics import mean, median
from time import perf_counter

from app.cache import get_cache_client, make_cache_key
from app.rag_pipeline import ask_question, _vector_store


QUESTION = "Explain semantic caching."
CACHE_RUNS = 10


def main():
    cache = get_cache_client()
    key = make_cache_key(QUESTION)

    try:
        # Ensure the first measured request is a cache miss.
        cache.delete(key)

        print("Running uncached request...")
        start = perf_counter()
        miss_response = ask_question(QUESTION)
        miss_ms = (perf_counter() - start) * 1000

        if miss_response["metrics"]["cache_hit"]:
            raise RuntimeError("Expected a cache miss.")

        print(f"Cache miss: {miss_ms:.2f} ms")

        # Measure repeated requests to the same question.
        hit_times = []

        print(f"\nRunning {CACHE_RUNS} cached requests...")

        for run in range(CACHE_RUNS):
            start = perf_counter()
            response = ask_question(QUESTION)
            elapsed_ms = (perf_counter() - start) * 1000

            if not response["metrics"]["cache_hit"]:
                raise RuntimeError(
                    f"Run {run + 1} was not a cache hit."
                )

            hit_times.append(elapsed_ms)
            print(f"Run {run + 1}: {elapsed_ms:.2f} ms")

        hit_mean = mean(hit_times)
        hit_median = median(hit_times)
        speedup = miss_ms / hit_median if hit_median else float("inf")
        reduction = (1 - hit_median / miss_ms) * 100

        print("\n===== CACHE BENCHMARK =====")
        print(f"Question: {QUESTION}")
        print(f"Cache miss: {miss_ms:.2f} ms")
        print(f"Cache hit mean: {hit_mean:.2f} ms")
        print(f"Cache hit median: {hit_median:.2f} ms")
        print(f"Cache hit min: {min(hit_times):.2f} ms")
        print(f"Cache hit max: {max(hit_times):.2f} ms")
        print(f"Median-based speedup: {speedup:.2f}x")
        print(f"Median latency reduction: {reduction:.2f}%")

    finally:
        cache.delete(key)
        cache.close()
        _vector_store.close()


if __name__ == "__main__":
    main()
