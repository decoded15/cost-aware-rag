from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_semantic_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_semantic_question(
    question: str,
    model: SentenceTransformer,
) -> list[float]:
    embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    return embedding.tolist()
