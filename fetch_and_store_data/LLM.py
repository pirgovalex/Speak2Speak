import os
from huggingface_hub import InferenceClient
from hybrid_search import hybrid_search
import platform

# Use Hugging Face Inference API for remote model access (no local GPU needed)
# Set your HF_TOKEN environment variable or replace with your token
HF_TOKEN = os.environ.get("HF_TOKEN", None)
# Using Qwen 2.5 72B Instruct - SOTA open model
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"


client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)


def clean_output(text):
    items = [m.strip() for m in text.split(',')]
    seen = set()
    unique_items = []
    for m in items:
        if m and m not in seen:
            unique_items.append(m)
            seen.add(m)
    return ', '.join(unique_items)


def llama_interact(q):
#mistral BIBLE
    prime_text = (
        "You are a highly precise medical assistant AI. Your goal is to answer questions using ONLY the provided context.\n\n"
        "### STRICT INSTRUCTIONS ###\n"
        "1. **Content Source**: You must derive your answer ENTIRELY from the 'Context' provided below. Do not use outside knowledge.\n"
        "2. **Output Format**: \n"
        "   - By default, provide a **clean, comma-separated list** of items (e.g., muscle names, bone names) without numbering or bullets.\n"
        "   - **EXCEPTION**: If the user explicitly asks for a 'DESCRIPTION' or 'EXPLANATION', provide a concise summary based on the text.\n"
        "3. **Negative Constraint**: \n"
        "   - Do NOT add chatty conversational filler (e.g., 'Here is the list:', 'Sure!').\n"
        "   - Do NOT invent information. If the answer is not in the context, reply exactly: 'Information not found in the documents.'\n"
        "4. **Hallucination Check**: Verify that every item in your list is explicitly mentioned in the text.\n\n"
    )
    
    docs = hybrid_search(q)
    context = "\n".join([doc.page_content for doc in docs])
    # Mistral Instruct/Zephyr is a chat model
    # We simplify the user prompt to prevent the model from generating its own conversation history
    messages = [
        {"role": "system", "content": prime_text},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=5000, 
        temperature=0.01,
        top_p=0.9,
    )
    
    # Extract text from the response
    output = response.choices[0].message.content
    
    # Aggressive cleaning: Stop if it tries to start a new turn
    stop_markers = ["Context:", "Question:", "User:", "Assistant:", "\n"]
    for marker in stop_markers:
        if marker in output:
            output = output.split(marker)[0]
    
    result = clean_output(output)
    print(result)
    return result