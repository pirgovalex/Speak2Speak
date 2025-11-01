from langchain_community.document_loaders import PyPDFLoader

file_path = "/home/aleksp/work/Greech/fetch&store_data/anatomy.pdf"
def load_pdf(file_path: str)-> list:
    loader = PyPDFLoader(file_path)
    pages = []
    pages = loader.load()
    return pages