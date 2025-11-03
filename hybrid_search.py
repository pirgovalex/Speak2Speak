from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATABASE = "faiss_index"
FILE_PATH = "anatomy.pdf"

def get_pdf() -> list:
    # Necessary for my hybrid search implementation.
    # This will consume more power but accuracy is key here.
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load()
    return pages

def get_files():
    possible_paths = ["faiss_index", "../faiss_index", "../../faiss_index"]
    for path in possible_paths:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("FAISS folder not found.")


def hybrid_search(q:str):
    folder = get_files()
    pages = get_pdf()
    embedding = HuggingFaceEmbeddings(model_name="iris49/3gpp-embedding-model-v0")
    faiss_db = FAISS.load_local(folder, embeddings=embedding,allow_dangerous_deserialization=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    docs = splitter.split_documents(pages)
    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 8

    ensemble = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever],weights=[0.5,0.6])


    result = ensemble.get_relevant_documents(q)
    return result

