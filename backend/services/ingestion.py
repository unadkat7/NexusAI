import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Initialize a text splitter: 500 characters per chunk with 50 characters overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

def parse_pdf(file_path: str, filename: str) -> list[Document]:
    """Parse a PDF file page by page into LangChain Document objects."""
    reader = PdfReader(file_path)
    documents = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            # Create a document for each non-empty page
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": i + 1,
                    "file_path": file_path
                }
            )
            documents.append(doc)
            
    return documents

def parse_text_file(file_path: str, filename: str) -> list[Document]:
    """Parse a text or markdown file into a LangChain Document object."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        
    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "file_path": file_path
            }
        )
    ]

def ingest_document(file_path: str, filename: str) -> list[Document]:
    """
    Ingests a file (PDF, TXT, MD, etc.), extracts text, 
    splits it into chunks, and returns a list of Document chunks.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        raw_docs = parse_pdf(file_path, filename)
    else:
        # Treats .txt, .md, .py, .js, .csv, etc. as plain text
        raw_docs = parse_text_file(file_path, filename)
        
    # Split raw documents into smaller chunks for vector & keyword indexing
    chunks = text_splitter.split_documents(raw_docs)
    
    # Add chunk_id to metadata for easy reference
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        
    return chunks
