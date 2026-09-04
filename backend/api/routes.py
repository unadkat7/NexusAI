import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from services.ingestion import ingest_document
from services.hybrid_retriever import hybrid_retriever
from db.database import get_db_connection

router = APIRouter(prefix="/api", tags=["Documents & Search"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

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
