
from time import perf_counter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODELS = {
    "MiniLM baseline": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE candidate": "BAAI/bge-small-en-v1.5",
}

REFERENCE_QUESTION = "How can I reduce repeated LLM calls?"

TEST_CASES = [
    (
        "exact",
        "How can I reduce repeated LLM calls?",
    ),
    (
        "paraphrase",
        "How do I avoid calling an LLM again for the same question?",
    ),
    (
        "paraphrase",
        "How can caching prevent repeated language model requests?",
    ),
    (
        "related",
        "How can I reduce the cost of repeated AI responses?",
    ),
    (
        "unrelated",
        "What is the capital of France?",
    ),
]


for model_label, model_name in MODELS.items():
    print(f"\n{'=' * 60}")
    print(f"Model: {model_label} ({model_name})")
    print("=" * 60)

    model = SentenceTransformer(model_name)

    start = perf_counter()
    reference_embedding = model.encode(
        [REFERENCE_QUESTION],
        normalize_embeddings=True,
    )
    reference_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    test_embeddings = model.encode(
        [question for _, question in TEST_CASES],
        normalize_embeddings=True,
    )
    test_ms = (perf_counter() - start) * 1000

    scores = cosine_similarity(
        test_embeddings,
        reference_embedding,
    ).flatten()

    for (category, question), score in zip(TEST_CASES, scores):
        print(f"\nCategory: {category}")
        print(f"Question: {question}")
        print(f"Similarity: {score:.4f}")

    print(f"\nReference embedding time: {reference_ms:.2f} ms")
    print(f"Batch test embedding time: {test_ms:.2f} ms")

    del model

print("\nEvaluation complete.")
