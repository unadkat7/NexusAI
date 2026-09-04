import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    PROJECT_NAME: str = "NexusAI"
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "nexus.db")
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "chroma_db")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()
