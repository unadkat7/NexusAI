import os
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote

# Load environment variables from .env file
load_dotenv()

def _sqlite_path_from_env() -> str:
    database_path = os.getenv("DATABASE_PATH")
    if database_path:
        return database_path

    database_url = os.getenv("DATABASE_URL", "")
    if database_url.startswith("sqlite:///"):
        parsed = urlparse(database_url)
        path = unquote(parsed.path)
        if path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        elif path.startswith("/./") or path.startswith("/../"):
            path = path[1:]
        return path or "nexus.db"

    return "nexus.db"

class Settings:
    PROJECT_NAME: str = "NexusAI"
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    DATABASE_PATH: str = _sqlite_path_from_env()
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "chroma_db")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()
