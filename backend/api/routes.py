import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.ingestion import ingest_document
from db.database import get_db_connection

router = APIRouter(prefix="/api", tags=["Documents & Chat"])

# Ensure uploads directory exists
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF, Markdown, or Text file.
    Parses and chunks the file, then records it in SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Ingest and chunk document
        chunks = ingest_document(file_path, file.filename)
        
        # Save record in SQLite
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
            "message": f"Successfully ingested {file.filename}",
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
