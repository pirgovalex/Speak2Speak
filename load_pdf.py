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
    model = SentenceTransformerEmbeddings(model_name="iris49/3gpp-embedding-model-v0")
    with open("faiss_index/docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    vector_db = FAISS.from_documents(docs, model)
    vector_db.save_local("faiss_index")

    print('SAVED!')






