
import statistics
from time import perf_counter

from app.rag_pipeline import ask_question, _vector_store

QUESTION = "What is semantic caching?"
RUNS = 5


def main():
    latencies = []

    try:
        for run in range(1, RUNS + 1):
            start = perf_counter()
            result = ask_question(QUESTION)
            elapsed_ms = (perf_counter() - start) * 1000

            latencies.append(elapsed_ms)

            print(
                f"Run {run}: {elapsed_ms / 1000:.2f}s"
            )
            print(f"Answer: {result['answer']}\n")

        print("Baseline results")
        print(f"Runs: {len(latencies)}")
        print(f"Mean: {statistics.mean(latencies) / 1000:.2f}s")
        print(f"Median: {statistics.median(latencies) / 1000:.2f}s")
        print(f"Min: {min(latencies) / 1000:.2f}s")
        print(f"Max: {max(latencies) / 1000:.2f}s")

    finally:
        _vector_store.close()


if __name__ == "__main__":
    main()
