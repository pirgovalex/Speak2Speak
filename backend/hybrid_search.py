import pickle
import os
import torch

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "..", "data", "faiss_index")

def _get_folder() -> str:
    if os.path.isdir(FAISS_DIR):
        return FAISS_DIR
    raise FileNotFoundError("faiss folder not found at " + FAISS_DIR)


_folder = _get_folder()

# remind: if the faiss index is rebuilt at runtime (e.g. new pdf uploaded),
# the server needs to restart for these module-level caches to reflect it
_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

_faiss_db = FAISS.load_local(
    _folder,
    embeddings=_embedding,
    allow_dangerous_deserialization=True,
)

with open(os.path.join(_folder, "docs.pkl"), "rb") as f:
    _docs = pickle.load(f)

_bm25_retriever = BM25Retriever.from_documents(_docs)
_bm25_retriever.k = 20

_cross_encoder = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

print(f"[hybrid_search] initialized on {DEVICE}")


def hybrid_search(q: str):
    faiss_retriever = _faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20},
    )
    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, _bm25_retriever],
        weights=[0.5, 0.5],
    )
    result = ensemble.get_relevant_documents(q)

    # rerank with cross encoder
    pairs = [[q, doc.page_content] for doc in result]
    scores = _cross_encoder.predict(pairs)

    scored_docs = sorted(zip(result, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs][:5]
