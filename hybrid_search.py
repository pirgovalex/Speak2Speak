import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os
from load_pdf import get_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

DATABASE = "faiss_index"

# Load models once at module level, not on every search call
_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_reranker = CrossEncoder("BAAI/bge-reranker-base")

def get_files()-> None:
    possible_paths = ["faiss_index", "../faiss_index", "../../faiss_index"]
    for path in possible_paths:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("FAISS folder not found.")


def hybrid_search(q:str):
    folder = get_files()
    faiss_db = FAISS.load_local(folder, embeddings=_embedding, allow_dangerous_deserialization=True)
    with open(f"{folder}/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    faiss_retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 20

    ensemble = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever],weights=[0.5,0.5])
    result = ensemble.get_relevant_documents(q)

    # rerank results with cross encoder
    pairs = [[q, doc.page_content] for doc in result]
    scores = _reranker.predict(pairs)

    # sort by score and get top 5
    scored_docs = list(zip(result, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    result = [doc for doc, score in scored_docs][:5]

    return result
