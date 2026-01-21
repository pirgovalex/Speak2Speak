import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from LLM import search_anatomy_tool

def test_page_numbers():
    print("--- Test Page Numbers ---")
    query = "muscle"
    result = search_anatomy_tool(query)
    print("Result Snippet:\n", result[:500])
    
    if "[Page" in result:
        print("\nSUCCESS: Page numbers found.")
    else:
        print("\nFAILURE: Page numbers NOT found.")

if __name__ == "__main__":
    test_page_numbers()
