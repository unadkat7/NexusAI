import sqlite3
import os
import json
from config import settings

def get_db_connection():
    """Connect to SQLite and enable WAL (Write-Ahead Logging) mode for performance."""
    db_dir = os.path.dirname(settings.DATABASE_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

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

def create_session_if_missing(session_id: str, title: str):
    """Create a chat session if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)",
        (session_id, title)
    )
    conn.commit()
    conn.close()

def add_message(session_id: str, role: str, content: str, sources: list[dict] | None = None):
    """Persist one chat message and optional source citations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources or []))
    )
    conn.commit()
    conn.close()

def get_recent_messages(session_id: str, limit: int = 8) -> list[dict]:
    """Return recent messages in chronological order for prompt memory."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content, sources, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    messages = []
    for row in reversed(rows):
        message = dict(row)
        message["sources"] = json.loads(message["sources"] or "[]")
        messages.append(message)

    return messages
