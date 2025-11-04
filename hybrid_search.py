import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from load_pdf import get_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATABASE = "faiss_index"

def get_files()-> None:
    possible_paths = ["faiss_index", "../faiss_index", "../../faiss_index"]
    for path in possible_paths:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("FAISS folder not found.")


def hybrid_search(q:str):
    folder = get_files()
    embedding = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
    faiss_db = FAISS.load_local(folder, embeddings=embedding,allow_dangerous_deserialization=True)
    with open("faiss_index/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    ensemble = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever],weights=[0.5,0.5])
    print(faiss_db.embedding_function)
    result = ensemble.get_relevant_documents(q)
    print("RELEVANT DOCUMENTS: ---------------------------------------------\n")
    print(result)
    print("END: ---------------------------------------------\n")
    return result

