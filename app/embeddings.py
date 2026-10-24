from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(
    chunks: list[str],
    model: SentenceTransformer,
) -> list[list[float]]:
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()
