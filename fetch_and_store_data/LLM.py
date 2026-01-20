import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from hybrid_search import hybrid_search
import platform

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

# Using Qwen 2.5 72B Instruct/GPT OSS 20B local - SOTA open model (via local endpoint)
llm = ChatOpenAI(
    temperature=0, 
    api_key="empty", 
    base_url="http://192.168.0.45:8008/v1", 
    model="openai/gpt-oss-20b"
)

# 1. Tool-ify the Knowledge Base
def search_anatomy_tool(query: str) -> str:
    """Useful for looking up anatomical details, muscle names, and medical facts."""
    docs = hybrid_search(query)
    # Return a single string of context
    return "\n\n".join([doc.page_content for doc in docs])

tools = [
    Tool(
        name="SearchAnatomyDocs",
        func=search_anatomy_tool,
        description="Useful for looking up anatomical details, muscle names, and medical facts. Input should be a specific search query."
    )
]

# 2. Define the System Prompt
system_prompt = '''You are a highly precise medical assistant AI. Your goal is to answer questions using ONLY the provided context tools.

### STRICT INSTRUCTIONS ###
1. **Content Source**: You must derive your answer ENTIRELY from the 'SearchAnatomyDocs' tool.
2. **Output Format**:
   - By default, provide a **clean, comma-separated list** of items (e.g., muscle names, bone names) without numbering or bullets.
   - **EXCEPTION**: If the user explicitly asks for a particular format (e.g. 'grouped by location', 'description'), follow their request.
3. **Negative Constraint**: 
   - Do NOT add chatty conversational filler (e.g., 'Here is the list:', 'Sure!').
   - Do NOT respond to anatomy questions that are different than human's anatomy.
   - Do NOT invent information. If the answer is not in the context, reply exactly: 'Information not found in the documents.'
'''

# 3. Create the Agent (LangGraph)
# create_react_agent returns a compiled graph
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

def llama_interact(q):
    try:
        # Run the agent (LangGraph invokes with state dict)
        response = agent_executor.invoke({"messages": [("user", q)]})
        
        # Extract final response from the last AI message
        result = response["messages"][-1].content
        
        # Post-processing cleanup (legacy constraint)
        if "Final Answer:" in result:
             result = result.split("Final Answer:")[-1].strip()
             
        print(result)
        return result
    except Exception as e:
        print(f"Error communicating with Agent: {e}")
        return "Error: Could not retrieve answer."