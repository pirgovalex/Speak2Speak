from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
import uvicorn
import threading

# Import internal modules
from LLM import llama_interact
import database_manager
import tts

app = FastAPI(title="Speak2Speak Chat API")

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class TTSRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class ThreadInfo(BaseModel):
    id: str

class Message(BaseModel):
    sender: str
    content: str
    timestamp: str

# Endpoints

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/threads/new", response_model=ThreadInfo)
async def create_thread():
    new_id = str(uuid.uuid4())[:8]
    # Initialize thread in DB with a placeholder system message
    try:
        database_manager.save_message(new_id, "system", "Thread created")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize thread: {e}")
    return ThreadInfo(id=new_id)

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, background_tasks: BackgroundTasks):
    # We return success immediately and delete in background
    # Note: validation of thread existence is skipped for speed, or we can check sync first
    try:
        background_tasks.add_task(database_manager.delete_thread, thread_id)
        return {"status": "success", "id": thread_id, "mode": "async"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def trigger_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(tts.speak, request.text)
        return {"status": "success", "message": "TTS queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/stop")
async def stop_tts():
    try:
        tts.stop()
        return {"status": "success", "message": "TTS stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/pause")
async def pause_tts():
    try:
        tts.pause()
        return {"status": "success", "message": "TTS paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/resume")
async def resume_tts():
    try:
        tts.resume()
        return {"status": "success", "message": "TTS resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads", response_model=List[str])
async def get_threads():
    try:
        threads = database_manager.get_recent_threads(limit=20)
        return threads
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/history", response_model=List[Message])
async def get_thread_history(thread_id: str):
    try:
        history = database_manager.get_chat_history(thread_id)
        # Convert to Pydantic models. database_manager returns list of dicts.
        # Ensure timestamp is string if it's not.
        clean_history = []
        for msg in history:
            clean_history.append(Message(
                sender=msg['sender'],
                content=msg['content'],
                timestamp=str(msg.get('timestamp', ''))
            ))
        return clean_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # LLM interaction can be slow, might block the event loop if not careful.
        # llama_interact is synchronous. We should run it in a threadpool or make it async.
        # FastAPI handles sync def endpoints by running them in a threadpool automatically.
        # So 'async def' might actually BLOCK if we call sync code directly.
        # BUT 'llama_interact' seems to do DB ops and LLM calls.
        
        # Best practice for sync blocking functions in FastAPI is to define the endpoint as 'def' (not async)
        # OR use await run_in_threadpool.
        # Let's switch this endpoint to 'def' so FastAPI runs it in a separate thread.
        return chat_sync(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def chat_sync(request: ChatRequest) -> ChatResponse:
    # This runs in a threadpool
    response_text = llama_interact(request.message, thread_id=request.thread_id)
    return ChatResponse(response=response_text, thread_id=request.thread_id)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
