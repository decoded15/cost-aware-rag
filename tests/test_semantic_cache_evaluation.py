
from app.semantic_embeddings import (
    load_semantic_embedding_model,
    embed_semantic_question,
)
from app.semantic_safety import has_conflicting_intent

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

QUESTION_PAIRS = [
    ("How can I reduce repeated LLM calls?",
     "How do I avoid calling an LLM again for the same question?", True),
    ("How can I reduce repeated LLM calls?",
     "How can caching prevent repeated language model requests?", True),
    ("How can I reduce the cost of repeated AI responses?",
     "How can I lower expenses from repeated LLM responses?", True),
    ("What is the purpose of semantic caching?",
     "Why use semantic caching in a RAG application?", True),
    ("How do I store vectors in Qdrant?",
     "How can I save embeddings in a vector database?", True),
    ("How do I create a FastAPI endpoint?",
     "How can I expose an API route using FastAPI?", True),
    ("How do I start Ollama?",
     "How can I launch the Ollama service?", True),
    ("How can I reduce LLM costs?",
     "How can I improve LLM answer accuracy?", False),
    ("How do I cache an answer?",
     "How do I retrieve documents from a vector database?", False),
    ("How do I delete expired cache entries?",
     "How do I extend cache entry expiration?", False),
    ("How do I start Ollama?",
     "How do I stop Ollama?", False),
    ("How do I create a FastAPI endpoint?",
     "How do I deploy a Flask application?", False),
    ("How can I reduce LLM costs?",
     "How do I train a neural network?", False),
    ("What is the capital of France?",
     "How do I configure a Qdrant collection?", False),
    ("How do I store vectors in Qdrant?",
     "How do I calculate a person's BMI?", False),
    ("How do I start Ollama?",
     "How do I restart Ollama after changing its configuration?", False),
    ("How do I increase the cache TTL?",
     "How do I reduce the cache TTL?", False),
    ("How do I enable semantic caching?",
     "How do I disable semantic caching?", False),
    ("How do I add a document to Qdrant?",
     "How do I delete a document from Qdrant?", False),
    ("How do I switch to a smaller language model?",
     "How do I switch to a larger language model?", False),
    ("How can I reduce LLM latency?",
     "How can I increase LLM throughput?", False),
    ("How do I configure a read timeout?",
     "How do I configure a connection timeout?", False),
    ("How can I improve LLM answer accuracy?",
     "How can I reduce LLM answer accuracy?", False),
]


def main():
    model = load_semantic_embedding_model()

    questions = list(dict.fromkeys(
        q for first, second, _ in QUESTION_PAIRS
        for q in (first, second)
    ))

    embeddings = {
        q: embed_semantic_question(q, model)
        for q in questions
    }

    evaluated = []
    for first, second, same_intent in QUESTION_PAIRS:
        score = sum(
            a * b for a, b in zip(embeddings[first], embeddings[second])
        )
        conflict = has_conflicting_intent(first, second)
        evaluated.append((same_intent, score, conflict, first, second))

    print(f"\nEvaluated {len(evaluated)} labeled question pairs.")
    print("\nTHRESHOLD COMPARISON")
    print(
        f"{'Threshold':<10} {'Method':<18} {'Correct hits':<14} "
        f"{'False hits':<12} {'Missed matches':<16} {'Correct misses'}"
    )

    for threshold in THRESHOLDS:
        for use_safety in (False, True):
            correct_hits = false_hits = missed = correct_misses = 0

            for same_intent, score, conflict, _, _ in evaluated:
                predicted_hit = score >= threshold
                if use_safety and conflict:
                    predicted_hit = False

                if same_intent and predicted_hit:
                    correct_hits += 1
                elif not same_intent and predicted_hit:
                    false_hits += 1
                elif same_intent:
                    missed += 1
                else:
                    correct_misses += 1

            method = "BGE + safety" if use_safety else "BGE only"
            print(
                f"{threshold:<10.2f} {method:<18} {correct_hits:<14} "
                f"{false_hits:<12} {missed:<16} {correct_misses}"
            )

    print("\nNEAR-MISS REVIEW")
    for same_intent, score, conflict, first, second in evaluated:
        if not same_intent:
            print(
                f"score={score:.4f} | conflict={conflict} | "
                f"{first!r} <> {second!r}"
            )


if __name__ == "__main__":
    main()
