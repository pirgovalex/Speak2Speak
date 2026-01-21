import sys
import os
import unittest
import time
import shutil

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database_manager
from LLM import llama_interact

TEST_DB = "integration_test_chats.db"

# Mock database_manager's DB_PATH dynamically for this test process would be ideal, 
# but database_manager uses a default arg. We can pass the path to save_message,
# but LLM.py calls it with default path.
# So we must modify LLM.py to accept db_path or mock it? 
# Or we just use a separate test db by monkeypatching?

# Let's monkeypatch database_manager.DB_PATH
database_manager.DB_PATH = TEST_DB

class TestIntegration(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        database_manager.init_db(TEST_DB)

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def test_llm_interaction_persistence(self):
        thread_id = "integration_thread_1"
        
        print("Sending message to LLM...")
        # We need to mock hybrid_search to avoid actual retrieving if possible, 
        # or just let it run if it's fast enough. 
        # For this environment, it might default to 'Information not found' which is fine.
        
        response = llama_interact("Hello, are you there?", thread_id=thread_id)
        
        print(f"LLM Response: {response}")
        
        # Check DB
        history = database_manager.get_chat_history(thread_id, TEST_DB)
        
        # Expecting 2 messages: User + AI
        self.assertEqual(len(history), 2)
        
        self.assertEqual(history[0]['sender'], 'user')
        self.assertEqual(history[0]['content'], "Hello, are you there?")
        
        self.assertEqual(history[1]['sender'], 'ai')
        self.assertEqual(history[1]['content'], response)
        
        # Check Thread List
        threads = database_manager.get_recent_threads(db_path=TEST_DB)
        self.assertIn(thread_id, threads)

if __name__ == '__main__':
    unittest.main()
