import sqlite3
import os
from config import settings

def get_db_connection():
    """Connect to SQLite and enable WAL (Write-Ahead Logging) mode for performance."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name: row['title']
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    """Initialize database tables for sessions, chat messages, and ingested documents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Sessions table (chat threads)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 2. Messages table (chat history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,  -- JSON string of citation sources
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
    """)
    
    # 3. Documents table (track uploaded files & indexed repos)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,  -- 'file' or 'github'
            chunk_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully in WAL mode.")
