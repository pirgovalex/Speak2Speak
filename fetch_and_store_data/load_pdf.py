import pickle
import os
import platform
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

if platform.system()=="Linux":
    FILE_PATH = "/home/aleksp/work/Speak2Speak/fetch_and_store_data/anatomy.pdf"
else:
    FILE_PATH = "anatomy.pdf"
def get_pdf()->list:   #Necessary for my hybrid search implementation.
                       #This will consume more power but accuracy is key here.
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load()
    return pages

def load_and_store_pdf()-> None:
    # Always allow running, even if folder exists
    pages = get_pdf()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                              separators=["\n\n", "\n", ".", " ", ""])
    docs = splitter.split_documents(pages)
    
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create the directory if it doesn't exist
    os.makedirs("faiss_index", exist_ok=True)
    
    # Save raw docs to docs.pkl to avoid collision with FAISS index.pkl
    with open("faiss_index/docs.pkl", "wb") as f:
        pickle.dump(docs, f)
        
    vector_db = FAISS.from_documents(docs, model)
    vector_db.save_local("faiss_index")

    print('SAVED!')






