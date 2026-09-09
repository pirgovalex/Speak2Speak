import pickle
import os
import threading

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FILE_PATH = os.path.join(DATA_DIR, "anatomy.pdf")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")


def get_pdf()->list:   #Necessary for my hybrid search implementation.
                       #This will consume more power but accuracy is key here.
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load()
    return pages

def load_and_store_pdf()-> None:
    if os.path.exists(FAISS_DIR) and len(os.listdir(FAISS_DIR)) > 0:
        return
    pages = get_pdf()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,
                                              separators=["\n\n", "\n", ".", " ", ""])
    docs = splitter.split_documents(pages)
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    os.makedirs(FAISS_DIR, exist_ok=True)
    with open(os.path.join(FAISS_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    vector_db = FAISS.from_documents(docs, model)
    vector_db.save_local(FAISS_DIR)

    print('SAVED!')

def add_pdf_to_index(file_path: str) -> None:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,
                                              separators=["\n\n", "\n", ".", " ", ""])
    new_docs = splitter.split_documents(pages)
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    os.makedirs(FAISS_DIR, exist_ok=True)
    if os.path.exists(FAISS_DIR) and len(os.listdir(FAISS_DIR)) > 0:
        faiss_db = FAISS.load_local(FAISS_DIR, embeddings=model, allow_dangerous_deserialization=True)
        new_db = FAISS.from_documents(new_docs, model)
        faiss_db.merge_from(new_db)
        faiss_db.save_local(FAISS_DIR)
        
        with open(os.path.join(FAISS_DIR, "docs.pkl"), "rb") as f:
            existing_docs = pickle.load(f)
        existing_docs.extend(new_docs)
        with open(os.path.join(FAISS_DIR, "docs.pkl"), "wb") as f:
            pickle.dump(existing_docs, f)
    else:
        with open(os.path.join(FAISS_DIR, "docs.pkl"), "wb") as f:
            pickle.dump(new_docs, f)
        vector_db = FAISS.from_documents(new_docs, model)
        vector_db.save_local(FAISS_DIR)
