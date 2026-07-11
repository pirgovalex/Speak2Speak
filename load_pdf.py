import pickle

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import threading


FILE_PATH = "anatomy.pdf"
def get_pdf()->list:   #Necessary for my hybrid search implementation.
                       #This will consume more power but accuracy is key here.
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load()
    return pages

def load_and_store_pdf()-> None:
    if "faiss_index" in os.listdir():
        return
    pages = get_pdf()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,
                                              separators=["\n\n", "\n", ".", " ", ""])
    docs = splitter.split_documents(pages)
    model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    with open("faiss_index/docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    vector_db = FAISS.from_documents(docs, model)
    vector_db.save_local("faiss_index")

    print('SAVED!')

def add_pdf_to_index(file_path: str) -> None:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,
                                              separators=["\n\n", "\n", ".", " ", ""])
    new_docs = splitter.split_documents(pages)
    model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index"):
        faiss_db = FAISS.load_local("faiss_index", embeddings=model, allow_dangerous_deserialization=True)
        new_db = FAISS.from_documents(new_docs, model)
        faiss_db.merge_from(new_db)
        faiss_db.save_local("faiss_index")
        
        with open("faiss_index/docs.pkl", "rb") as f:
            existing_docs = pickle.load(f)
        existing_docs.extend(new_docs)
        with open("faiss_index/docs.pkl", "wb") as f:
            pickle.dump(existing_docs, f)
    else:
        os.makedirs("faiss_index", exist_ok=True)
        with open("faiss_index/docs.pkl", "wb") as f:
            pickle.dump(new_docs, f)
        vector_db = FAISS.from_documents(new_docs, model)
        vector_db.save_local("faiss_index")
