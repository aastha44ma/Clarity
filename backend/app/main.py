from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.services.rag import ask_question, index_pdf

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="RAG-based AI Document and Policy Analyzer",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini_configured": bool(settings.google_api_key),
        "pinecone_configured": bool(settings.pinecone_api_key),
        "pinecone_index": settings.pinecone_index_name,
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        return index_pdf(file)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(error)}",
        )


@app.post("/api/chat")
async def chat(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return ask_question(request.question)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(error)}",
        )
