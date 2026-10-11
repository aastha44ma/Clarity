import uuid
import fitz
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.config import settings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.google_api_key,
)

pc = Pinecone(api_key=settings.pinecone_api_key)
vector_store = None

try:
    # Initialize with safe fallbacks for missing/unusable indexes on free tiers
    index = pc.Index(settings.pinecone_index_name)
    vector_store = PineconeVectorStore(
        embedding=embeddings,
        index=index,
    )
    print(f"Successfully connected to Pinecone index: {settings.pinecone_index_name}")
except Exception as e:
    print(f"[Warning]: Could not connect to Pinecone index '{settings.pinecone_index_name}'. Error: {e}")
    print("Backend will start, but vector storage features (upload/chat) will require index configuration.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0,
)

SYSTEM_PROMPT = """You are a precise AI document analysis assistant.

Answer ONLY using the document context provided below.

Rules:
1. If the answer is not present in the context, reply exactly:
   \"The uploaded documents do not contain enough information to answer this.\"
2. Cite the source filename for every factual claim.
3. Do not invent clauses, dates, penalties, obligations, definitions, or policies.
4. For legal, compliance, HR, or policy questions, add:
   \"This is not legal advice.\"
5. Be concise, structured, and easy to read.
"""


def extract_pdf_text(file: UploadFile) -> str:
    content = file.file.read()

    with fitz.open(stream=content, filetype="pdf") as document:
        pages = [page.get_text("text") for page in document]

    return "\n\n".join(pages)


def index_pdf(file: UploadFile) -> dict:
    if vector_store is None:
        raise ValueError(
            f"Pinecone index '{settings.pinecone_index_name}' is not initialized or accessible. "
            "Please check your index name and free tier limits."
        )

    document_id = str(uuid.uuid4())
    raw_text = extract_pdf_text(file)

    if not raw_text.strip():
        raise ValueError(
            "No selectable text found. This may be a scanned PDF requiring OCR."
        ) 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(raw_text)

    texts = []
    metadatas = []

    for chunk_index, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append(
            {
                "document_id": document_id,
                "filename": file.filename,
                "chunk_index": chunk_index,
            }
        )

    vector_store.add_texts(texts=texts, metadatas=metadatas)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }


def ask_question(question: str) -> dict:
    if vector_store is None:
        return {
            "answer": "Vector store index is not available. Please verify your Pinecone index config.",
            "sources": [],
        }

    retrieved_docs = vector_store.similarity_search(question, k=5)

    if not retrieved_docs:
        return {
            "answer": "No documents have been indexed yet. Please upload a PDF first.",
            "sources": [],
        }

    context_blocks = []

    for doc in retrieved_docs:
        filename = doc.metadata.get("filename", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "unknown")

        context_blocks.append(
            f"[Source: {filename} | Chunk: {chunk_index}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_PROMPT}

Question:
{question}

Document context:
{context}
"""

    response = llm.invoke(prompt)

    sources = [
        {
            "filename": doc.metadata.get("filename", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "snippet": doc.page_content[:400],
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
