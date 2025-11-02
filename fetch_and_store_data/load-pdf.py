from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

FILE_PATH = "anatomy.pdf"
def get_pdf()->list:   #Necessary for my hybrid search implementation.
                       #This will consume more power but accuracy is key here.
    loader = PyPDFLoader(FILE_PATH)
    pages = loader.load()
    return pages

def load_and_store_pdf(file_path:str)-> None:
    pages = get_pdf()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    docs = splitter.split_documents(pages)
    model = SentenceTransformerEmbeddings(model_name="iris49/3gpp-embedding-model-v0")
    vector_db = FAISS.from_documents(docs, model)
    vector_db.save_local("faiss_index")
    print('SAVED!')





if __name__ == '__main__':
    if "faiss_index" not in os.listdir():
        load_and_store_pdf(FILE_PATH)