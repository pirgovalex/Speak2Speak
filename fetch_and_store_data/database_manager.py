import sqlite3
import os
import datetime

DB_PATH = "chats.db"

def init_db(db_path=None):
    """Initializes the database with threads and messages tables."""
    if db_path is None:
        db_path = DB_PATH
        
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Table for storing conversation threads
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            name TEXT
        )
    ''')

    # Table for storing messages
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            sender TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(thread_id) REFERENCES threads(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_message(thread_id, sender, content, db_path=None):
    """Saves a message to the database. Creates thread if not exists."""
    if db_path is None:
        db_path = DB_PATH
        
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Check if thread exists, if not create it
    cursor.execute("SELECT id FROM threads WHERE id = ?", (thread_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO threads (id) VALUES (?)", (thread_id,))
    
    cursor.execute(
        "INSERT INTO messages (thread_id, sender, content) VALUES (?, ?, ?)",
        (thread_id, sender, content)
    )
    
    conn.commit()
    conn.close()

def get_chat_history(thread_id, db_path=None):
    """Retrieves all messages for a given thread."""
    if db_path is None:
        db_path = DB_PATH
        
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT sender, content, timestamp FROM messages WHERE thread_id = ? ORDER BY timestamp ASC",
        (thread_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [{"sender": row[0], "content": row[1], "timestamp": row[2]} for row in rows]

def get_recent_threads(limit=10, db_path=None):
    """Returns a list of recent thread IDs."""
    if db_path is None:
        db_path = DB_PATH
        
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Get threads ordered by latest message
    cursor.execute('''
        SELECT DISTINCT t.id 
        FROM threads t
        LEFT JOIN messages m ON t.id = m.thread_id
        ORDER BY m.timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [row[0] for row in rows]

# Initialize DB on import if not exists (or we can call it explicitly)
# Pass current global DB_PATH effectively
init_db()
