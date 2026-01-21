import unittest
import sys
import os
import sqlite3
import time

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database_manager

TEST_DB = "test_chats.db"

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Clean up previous test db
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        database_manager.init_db(TEST_DB)

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def test_save_and_retrieve_message(self):
        thread_id = "test_thread_1"
        database_manager.save_message(thread_id, "user", "Hello", TEST_DB)
        database_manager.save_message(thread_id, "ai", "Hi there", TEST_DB)
        
        history = database_manager.get_chat_history(thread_id, TEST_DB)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["sender"], "user")
        self.assertEqual(history[0]["content"], "Hello")
        self.assertEqual(history[1]["sender"], "ai")
        self.assertEqual(history[1]["content"], "Hi there")

    def test_get_recent_threads(self):
        thread1 = "thread_1"
        thread2 = "thread_2"
        
        database_manager.save_message(thread1, "user", "msg1", TEST_DB)
        time.sleep(1.1) 
        database_manager.save_message(thread2, "user", "msg2", TEST_DB)
        
        threads = database_manager.get_recent_threads(db_path=TEST_DB)
        self.assertIn(thread1, threads)
        self.assertIn(thread2, threads)
        # Check order (thread2 is more recent)
        self.assertEqual(threads[0], thread2)

if __name__ == '__main__':
    unittest.main()
