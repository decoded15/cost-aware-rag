from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import UploadFile, File
from app.ingestion_service import ingest_text
from app.rag_pipeline import _embedding_model, _vector_store
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

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported.",
        )

    try:
        content = await file.read()
        text = content.decode("utf-8")

        result = ingest_text(
            text=text,
            source=file.filename,
            embedding_model=_embedding_model,
            vector_store=_vector_store,
        )

        return {
            "status": "success",
            "message": "Document ingested successfully.",
            **result,
        }

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be a valid UTF-8 text file.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to ingest document.",
        ) from exc
    finally:
        await file.close()