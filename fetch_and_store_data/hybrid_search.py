import pickle
import os

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from load_pdf import get_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import platform

if platform.system()=="Windows":
    DATABASE = "faiss_index"
else:
    DATABASE = "/home/aleksp/work/Speak2Speak/fetch_and_store_data/faiss_index"
def get_files()-> None:
    possible_paths = [DATABASE, f"../{DATABASE}", f"../../{DATABASE}"]
    for path in possible_paths:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("FAISS folder not found.")


def hybrid_search(q:str):
    folder = get_files()
    # Explicitly use CPU for embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    faiss_db = FAISS.load_local(folder, embeddings=embedding, allow_dangerous_deserialization=True)
    with open(f"{folder}/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 20

    ensemble = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever],weights=[0.3,0.7])
    print(faiss_db.embedding_function)
    result = ensemble.invoke(q)
    print("RELEVANT DOCUMENTS: ---------------------------------------------\n")
    print(result)
    print("END: ---------------------------------------------\n")
    return result

