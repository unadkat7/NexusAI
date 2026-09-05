import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from services.ingestion import ingest_document
from services.hybrid_retriever import hybrid_retriever
from graph.workflow import rag_workflow
from db.database import (
    add_message,
    create_session_if_missing,
    get_db_connection,
    get_recent_messages,
)

router = APIRouter(prefix="/api", tags=["Documents & Search"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Which framework is used in background job queueing in CodeStage?",
                "top_k": 5
            }
        }
    )

    message: str
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD), chunk it, 
    index it into ChromaDB + BM25, and record in SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Parse & Chunk Document
        chunks = ingest_document(file_path, file.filename)
        
        # 2. Add chunks to Hybrid Retriever (ChromaDB + BM25)
        hybrid_retriever.add_documents(chunks)
        
        # 3. Save Document record in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (name, source_type, chunk_count) VALUES (?, ?, ?)",
            (file.filename, "file", len(chunks))
        )
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {file.filename}",
            "filename": file.filename,
            "chunks_count": len(chunks),
            "sample_chunk": chunks[0].page_content if chunks else ""
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.get("/documents")
def get_documents():
    """Retrieve list of all ingested documents from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, source_type, chunk_count, created_at FROM documents ORDER BY id DESC")
    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"documents": docs}

@router.delete("/documents/{document_id}")
def delete_document(document_id: int):
    """Delete an uploaded document record, file, and indexed chunks."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM documents WHERE id = ?", (document_id,))
    document = cursor.fetchone()

    if not document:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    filename = document["name"]
    deleted_chunks = hybrid_retriever.delete_documents_by_source(filename)

    cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
    conn.close()

    file_path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    file_deleted = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_deleted = True

    return {
        "status": "success",
        "message": f"Deleted {filename}",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
        "file_deleted": file_deleted,
    }

@router.post("/search")
def search_documents(request: SearchRequest):
    """
    Executes Hybrid RRF Search (Dense Vector + BM25 Keyword) 
    and returns top matching context chunks with metadata.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
        
    results = hybrid_retriever.hybrid_search(query=request.query, top_k=request.top_k)
    
    formatted_results = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]
    
    return {
        "query": request.query,
        "results_count": len(formatted_results),
        "results": formatted_results
    }

@router.post("/chat")
def chat(request: ChatRequest):
    """
    Run the Step 5 RAG chat workflow:
    session memory from SQLite -> hybrid retrieval -> Gemini answer -> persisted response.
    """
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    requested_session_id = (request.session_id or "").strip()
    session_id = (
        requested_session_id
        if requested_session_id and requested_session_id.lower() != "string"
        else str(uuid.uuid4())
    )
    title = question[:60] or "Untitled chat"

    try:
        create_session_if_missing(session_id, title)
        chat_history = get_recent_messages(session_id)
        add_message(session_id, "user", question)

        result = rag_workflow.invoke({
            "question": question,
            "chat_history": chat_history,
            "top_k": request.top_k,
            "documents": [],
            "answer": "",
            "sources": [],
        })

        answer = result["answer"]
        sources = result.get("sources", [])
        add_message(session_id, "assistant", answer, sources)

        return {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat workflow failed: {str(e)}")

@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Return recent messages for a chat session."""
    return {
        "session_id": session_id,
        "messages": get_recent_messages(session_id, limit=50),
    }
