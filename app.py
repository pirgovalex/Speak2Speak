from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from LLM import llama_interact
from pydantic import BaseModel
from load_pdf import add_pdf_to_index
import shutil
import os

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    pass

@app.post("/query")
def extract_query(req: QueryRequest):
    query = req.question
    answer = llama_interact(query)
    return {"answer": answer, "sources":[]}

@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    add_pdf_to_index(file_path)
    os.remove(file_path)
    return {"status": "success"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[" http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers= ["*"],
)