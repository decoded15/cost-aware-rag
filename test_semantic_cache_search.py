from app.embeddings import load_embedding_model
from app.vector_store import (
    create_vector_store,
    create_semantic_cache_collection,
    store_semantic_cache_entry,
    search_semantic_cache,
)

SIMILARITY_THRESHOLD = 0.90

model = load_embedding_model()
client = create_vector_store()


try:
    create_semantic_cache_collection(client)

    original_question = "How can I reduce repeated LLM calls?"

    original_embedding = model.encode(
        original_question,
        normalize_embeddings=True,
    ).tolist()

    response = {
        "answer": "Caching can avoid repeated generation.",
        "sources": [],
        "metrics": {},
    }

    store_semantic_cache_entry(
        client,
        original_question,
        original_embedding,
        response,
        ttl_seconds=600,
    )

    test_questions = [
        "How can I reduce repeated LLM calls?",
        "How can caching prevent repeated language model requests?",
        "How do I avoid calling an LLM again for the same question?",
        "How can I reduce the cost of repeated AI responses?",
        "What is the capital of France?",
    ]

    for test_question in test_questions:
        query_embedding = model.encode(
            test_question,
            normalize_embeddings=True,
        ).tolist()

        raw_results = client.query_points(
            collection_name="semantic_cache",
            query=query_embedding,
            limit=1,
            with_payload=True,
        ).points

        print(f"\nQuestion: {test_question}")

        if raw_results:
            nearest = raw_results[0]
            print(f"Nearest question: {nearest.payload['question']}")
            print(f"Raw similarity: {nearest.score:.4f}")
        else:
            print("Raw similarity: no candidates")

        match = search_semantic_cache(
            client,
            query_embedding,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        if match is None:
            print("Cache result: MISS")
        else:
            print(f"Cache result: HIT (score={match['score']:.4f})")
            print(f"Cached answer: {match['response']['answer']}")

    exact_match = search_semantic_cache(
        client,
        original_embedding,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    assert exact_match is not None
    assert exact_match["response"]["answer"] == response["answer"]
    assert exact_match["score"] >= SIMILARITY_THRESHOLD

    print("\nExact-match test: PASS")

finally:
    client.close()
