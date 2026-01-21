import sqlite3
import os
from langgraph.checkpoint.sqlite import SqliteSaver

def get_checkpointer(db_path="chat_history.db"):
    """
    Creates and returns a SqliteSaver checkpointer with a persistent connection.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Ensure the table is created? SqliteSaver usually does this.
    return SqliteSaver(conn)

def get_all_threads(db_path="chat_history.db"):
    """
    Returns a list of all unique thread_ids from the checkpoints table.
    """
    try:
        if not os.path.exists(db_path):
            return []
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
        if not cursor.fetchone():
            conn.close()
            return []

        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception as e:
        print(f"Error reading threads: {e}")
        return []
