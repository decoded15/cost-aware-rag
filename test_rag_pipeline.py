from app.rag_pipeline import ask_question, _vector_store

try:
    question = "How can we reduce the cost of repeated AI responses?"
    result = ask_question(question)

    print("\nQuestion:", question)
    print("\nAnswer:\n", result["answer"])
    print("\nRetrieved sources:")

    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(chunk {source['chunk_index']}, "
            f"score: {source['score']:.3f})"
        )
finally:
    _vector_store.close()