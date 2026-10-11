# AI Document & Policy Analyzer
**Deployed website:** https://clarity-umber-eta.vercel.app/
A full-stack Retrieval-Augmented Generation (RAG) application that lets users
upload PDF documents and ask context-based questions about them.

## Features
- Upload multiple PDF documents
- Extract and chunk PDF text
- Generate embeddings using Google Gemini
- Store vectors in Pinecone
- Answer questions using retrieved document context
- Show source citations and retrieved chunks
- Next.js frontend deployed on Vercel
- FastAPI backend with LangChain

## Tech Stack
- Next.js, React, TypeScript, Tailwind CSS
- FastAPI, Uvicorn
- LangChain, LangChain-Google-GenAI
- Pinecone Vector Database
- Google Gemini Embeddings and Chat Model
- PyMuPDF for PDF extraction
- ngrok for temporary backend exposure

## API
- `GET /health`
- `POST /api/documents/upload`
- `POST /api/chat`

## Environment Variables
### Backend
- GOOGLE_API_KEY
- PINECONE_API_KEY
- PINECONE_INDEX_NAME
- FRONTEND_ORIGINS

### Frontend
- NEXT_PUBLIC_API_URL

## Disclaimer
This tool provides document-based answers and is not legal advice.
