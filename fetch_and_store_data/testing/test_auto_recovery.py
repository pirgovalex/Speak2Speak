import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LLM
import database_manager

class TestAutoRecovery(unittest.TestCase):
    def setUp(self):
        # Mock database_manager to avoid side effects
        self.mock_db = patch('LLM.database_manager').start()
        
        # Mock agent_executor
        self.mock_executor = patch('LLM.agent_executor').start()

    def tearDown(self):
        patch.stopall()

    def test_auto_recovery_on_corruption(self):
        # Setup: First call raises corruption error, second call succeeds
        corruption_error = Exception("Found AIMessages with tool_calls that do not have a corresponding ToolMessage")
        success_response = {"messages": [MagicMock(content="Recovery Successful")]}
        
        self.mock_executor.invoke.side_effect = [corruption_error, success_response]
        
        # Execute
        thread_id = "corrupted_thread_1"
        response = LLM.llama_interact("Help me", thread_id=thread_id)
        
        # Verify
        print(f"Response: {response}")
        self.assertEqual(response, "Recovery Successful")
        
        # Verify allow_recovery was triggered (invoke called twice)
        self.assertEqual(self.mock_executor.invoke.call_count, 2)
        
        # Verify first call used original thread_id
        args1, kwargs1 = self.mock_executor.invoke.call_args_list[0]
        self.assertEqual(kwargs1['config']['configurable']['thread_id'], thread_id)
        
        # Verify second call used a different ID (recovery ID)
        args2, kwargs2 = self.mock_executor.invoke.call_args_list[1]
        recovery_id = kwargs2['config']['configurable']['thread_id']
        self.assertNotEqual(recovery_id, thread_id)
        self.assertIn(thread_id, recovery_id)
        self.assertIn("_recovery_", recovery_id)
        
        # Verify message was saved to ORIGINAL thread_id (for user continuity)
        # LLM.py saves "user" message, then "ai" message.
        # "user" msg saved twice? No, only once at start.
        # "ai" msg saved once at end.
        
        # Check save_message calls
        # 1. User message (original thread)
        # 2. AI message (original thread, despite internal recovery)
        
        self.mock_db.save_message.assert_any_call(thread_id, "user", "Help me")
        self.mock_db.save_message.assert_any_call(thread_id, "ai", "Recovery Successful")

if __name__ == '__main__':
    unittest.main()
