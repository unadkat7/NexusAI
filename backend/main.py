from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from db.database import init_db
from api.routes import router

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Self-Hosted AI Research Assistant API with Hybrid RAG & LangGraph",
    version="1.0.0"
)

# Enable CORS for Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Next.js frontend to communicate with backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routes
app.include_router(router)

@app.on_event("startup")
def startup_event():
    """Runs when the FastAPI server starts."""
    init_db()

@app.get("/api/health")
def health_check():
    """Simple health check endpoint to verify backend is running."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "database": "SQLite (WAL Mode) Ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)

