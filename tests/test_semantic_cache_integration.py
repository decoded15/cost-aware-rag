
from uuid import uuid4

from app.cache import get_cache_client, make_cache_key
from app.rag_pipeline import ask_question


def main():
    cache = get_cache_client()
    run_id = uuid4().hex[:8]

    question = (
        f"Explain why a cache avoids repeated model calls in software systems. "
        f"Reference {run_id}."
    )
    paraphrase = (
        f"Explain how storing previous outputs prevents repeated model requests. "
        f"Reference {run_id}."
    )

    try:
        # Ensure exact-cache entries cannot affect this run.
        cache.delete(make_cache_key(question))
        cache.delete(make_cache_key(paraphrase))

        first = ask_question(question)
        print("FIRST REQUEST")
        print("Metrics:", first["metrics"])

        second = ask_question(paraphrase)
        print("\nSECOND REQUEST")
        print("Metrics:", second["metrics"])

        if first["metrics"]["semantic_cache_hit"]:
            print("\nFirst request unexpectedly hit semantic cache.")
            print("Use a fresh run ID and try again.")
            return

        if second["metrics"]["semantic_cache_hit"]:
            print("\nSemantic-cache integration: PASSED")
        else:
            print("\nNo semantic hit for this paraphrase.")
            print("The cache may be working; this pair did not match.")

    finally:
        cache.close()


if __name__ == "__main__":
    main()
