
from app.semantic_safety import has_conflicting_intent


TEST_CASES = [
    (
        "How do I increase the cache TTL?",
        "How do I reduce the cache TTL?",
        True,
    ),
    (
        "How do I enable semantic caching?",
        "How do I disable semantic caching?",
        True,
    ),
    (
        "How do I start Ollama?",
        "How do I stop Ollama?",
        True,
    ),
    (
        "How do I add a document to Qdrant?",
        "How do I delete a document from Qdrant?",
        True,
    ),
    (
        "How do I switch to a smaller language model?",
        "How do I switch to a larger language model?",
        True,
    ),
    (
        "How do I start Ollama?",
        "How can I launch the Ollama service?",
        False,
    ),
    (
        "How do I create a FastAPI endpoint?",
        "How can I expose an API route using FastAPI?",
        False,
    ),
]


def main():
    passed = 0

    for first, second, expected in TEST_CASES:
        actual = has_conflicting_intent(first, second)
        success = actual == expected
        passed += success

        status = "PASS" if success else "FAIL"
        print(f"{status}: {first!r} vs. {second!r}")
        print(f"  Expected: {expected}, Actual: {actual}")

    print(f"\nPassed {passed}/{len(TEST_CASES)} tests.")

    if passed != len(TEST_CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
