from app.embeddings import load_embedding_model
from app.generation import generate_answer
from app.vector_store import create_vector_store, search_chunks

_embedding_model = load_embedding_model()
_vector_store = create_vector_store()


def ask_question(question: str) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    query_embedding = _embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = search_chunks(
        client=_vector_store,
        query_embedding=query_embedding,
        limit=3,
    )

    if not results:
        return {
            "answer": "I couldn't find relevant information in the stored documents.",
            "sources": [],
        }

    context_parts = []
    sources = []

    for result in results:
        source = result["source"]
        chunk_index = result["chunk_index"]
        text = result["text"]

        context_parts.append(
            f"[Source: {source}, chunk: {chunk_index}]\n{text}"
        )

        sources.append({
            "source": source,
            "chunk_index": chunk_index,
            "score": result["score"],
        })

    context = "\n\n".join(context_parts)

    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": sources,
    }
