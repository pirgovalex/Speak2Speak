import os
import platform
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from hybrid_search import hybrid_search
from db_utils import get_checkpointer
import database_manager

# Using Qwen 2.5 72B Instruct/GPT OSS 20B local - SOTA open model (via local endpoint)
llm = ChatOpenAI(
    temperature=0, 
    api_key="empty", 
    # base_url="http://192.168.0.45:8008/v1", 
    base_url="http://192.168.0.33:8000/v1", 
    # model="openai/gpt-oss-20b"
    model="/models/gpt-oss-120b"
)

# 1. Tool-ify the Knowledge Base
def search_anatomy_tool(query: str) -> str:
    """Useful for looking up anatomical details, muscle names, and medical facts."""
    docs = hybrid_search(query)
    
    # Format content with page numbers
    formatted_docs = []
    for doc in docs:
        page_num = doc.metadata.get("page", "Unknown")
        # Add 1 to page number because most PDFs interact users are 1-indexed, but PyPDFLoader is 0-indexed
        try:
           page_display = int(page_num) + 1
        except:
           page_display = page_num
           
        formatted_docs.append(f"[Page {page_display}]\n{doc.page_content}")
        
    return "\n\n---\n\n".join(formatted_docs)

tools = [
    Tool(
        name="SearchAnatomyDocs",
        func=search_anatomy_tool,
        description="Useful for looking up anatomical details, muscle names, and medical facts. Input should be a specific search query."
    )
]

system_prompt = '''You are a highly precise medical assistant AI. Your goal is to answer questions using ONLY the provided context tools, but you may recall conversation history for context.

### STRICT INSTRUCTIONS ###
1. **Content Source**: For medical facts, you must derive your answer ENTIRELY from the 'SearchAnatomyDocs' tool. For conversational context (e.g. user's name, previous questions), use chat history.
2. **Page References**: When you provide information, **YOU MUST** cite the page number(s) provided in the context (e.g. "[Page 12]"). If multiple pages are relevant, list them.
3. **Output Format**:
   - By default, provide a **clean, comma-separated list** of items (e.g., muscle names, bone names) without numbering or bullets, followed by the page citation.
   - **EXCEPTION**: If the user explicitly asks for a particular format (e.g. 'grouped by location', 'description'), follow their request.
    - **EXCEPTION**:  If user asks for a TEST - - use the necessary tools to find data for the test questions plus the page of the topic/section(s)
    , in whatever format the user requests. Prettify the output test.
4. **Negative Constraint**: 
   - Do NOT add chatty conversational filler (e.g., 'Here is the list:', 'Sure!').
   - Do NOT respond to anatomy questions that are different than human's anatomy.
   - Do NOT invent information. If the answer is not in the context, reply exactly: 'Information not found in the documents.'
'''

# 3. Create the Agent (LangGraph) w/ Persistence
memory = get_checkpointer("chat_history.db")
# create_react_agent returns a compiled graph
# Using messages_modifier or prompt depending on version. 
# Using messages_modifier or prompt depending on version. 
agent_executor = create_react_agent(llm, tools, prompt=system_prompt, checkpointer=memory)

def llama_interact(q, thread_id=None):
    if thread_id is None:
        # Generate a random one if not provided, or fix it to a default for single-user dev
        thread_id = "default_user_thread"

    # Save USER message to separate DB
    database_manager.save_message(thread_id, "user", q)

    # Config for the invocation
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run the agent (LangGraph invokes with state dict)
        response = agent_executor.invoke({"messages": [("user", q)]}, config=config)
        
        # Extract final response from the last AI message
        result = response["messages"][-1].content
        
        # Post-processing cleanup (legacy constraint)
        if "Final Answer:" in result:
             result = result.split("Final Answer:")[-1].strip()
        
        # Save AI message to separate DB
        database_manager.save_message(thread_id, "ai", result)
             
        print(result)
        return result
    except Exception as e:
        error_msg = str(e)
        # Check for the specific LangGraph state corruption error
        if "Found AIMessages with tool_calls that do not have a corresponding ToolMessage" in error_msg:
             print(f"Warning: Thread {thread_id} state corrupted. Attempting auto-recovery...")
             
             # Create a recovery thread ID to bypass the broken state
             recovery_id = f"{thread_id}_recovery_{str(uuid.uuid4())[:4]}"
             print(f"Switching internal context to recovery thread: {recovery_id}")
             
             # Retry with new ID
             try:
                config_recovery = {"configurable": {"thread_id": recovery_id}}
                response = agent_executor.invoke({"messages": [("user", q)]}, config=config_recovery)
                result = response["messages"][-1].content
                
                if "Final Answer:" in result:
                     result = result.split("Final Answer:")[-1].strip()
                
                # Save to ORIGINAL thread in DB so user sees continuity
                database_manager.save_message(thread_id, "ai", result)
                
                print(result)
                return result
             except Exception as e2:
                 print(f"Recovery failed: {e2}")
                 return f"Error: Session corrupted and recovery failed. Please start a New Chat."
        
        print(f"Error communicating with Agent: {e}")
        return f"Error: {e}"