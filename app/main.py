
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import ask_question

app = FastAPI(
    title="Cost-Aware RAG Gateway",
    description="A document-grounded question-answering API.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer using stored documents.",
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        return ask_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the question.",
        ) from exc
