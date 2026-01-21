import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from LLM import llama_interact

def test_persistence():
    print("--- Test 1: Set Name ---")
    # Use a unique thread_id for this test
    tid = "test_user_123"
    
    # 1. Tell the agent my name
    response1 = llama_interact("My name is TestUser.", thread_id=tid)
    print(f"Response 1: {response1}")
    
    print("\n--- Test 2: Recall Name ---")
    # 2. Ask the agent to recall the name
    # If persistence works, it should know the name
    response2 = llama_interact("What is my name?", thread_id=tid)
    print(f"Response 2: {response2}")
    
    if "TestUser" in response2:
        print("\nSUCCESS: Persistence working.")
    else:
        print("\nFAILURE: Persistence NOT working.")

if __name__ == "__main__":
    test_persistence()
