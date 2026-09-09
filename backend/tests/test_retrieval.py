import pytest
import os
from unittest.mock import MagicMock

# mock models to avoid downloading large models during tests
@pytest.fixture
def mock_embeddings(mocker):
    # mock both huggingfaceembeddings and sentencetransformerembeddings
    mocker.patch("hybrid_search.HuggingFaceEmbeddings")
    mocker.patch("load_pdf.HuggingFaceEmbeddings")

@pytest.fixture
def mock_cross_encoder(mocker):
    # mock cross encoder
    return mocker.patch("hybrid_search.CrossEncoder")

@pytest.fixture
def mock_faiss_db(mocker):
    # mock the module level _faiss_db instance directly
    return mocker.patch("hybrid_search._faiss_db")

@pytest.fixture
def mock_faiss_load(mocker):
    # mock faiss for load pdf
    return mocker.patch("load_pdf.FAISS")

def test_get_pdf_signature(mocker):
    # test function signature and mock loader
    mock_loader = mocker.patch("load_pdf.PyPDFLoader")
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load.return_value = ["page1", "page2"]
    
    from load_pdf import get_pdf
    result = get_pdf()
    
    # check if result is a list as per signature
    assert isinstance(result, list)
    assert result == ["page1", "page2"]

def test_load_and_store_pdf(mocker, mock_embeddings, mock_faiss_load):
    # mock os listdir to simulate fresh run
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.makedirs")
    
    # mock get pdf to return dummy pages
    mock_doc = MagicMock()
    mock_doc.page_content = "dummy text"
    mocker.patch("load_pdf.get_pdf", return_value=[mock_doc])
    
    # mock splitter
    mock_splitter = mocker.patch("load_pdf.RecursiveCharacterTextSplitter")
    mock_splitter_instance = mock_splitter.return_value
    mock_splitter_instance.split_documents.return_value = [mock_doc]
    
    # mock file operations
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("pickle.dump")
    
    from load_pdf import load_and_store_pdf
    load_and_store_pdf()
    
    # check if docs are saved
    # We can check that open was called with docs.pkl
    mock_open.assert_called()
    mock_faiss_load.from_documents.assert_called()

def test_get_folder(mocker):
    # test file path resolution
    mocker.patch("os.path.isdir", return_value=True)
    from hybrid_search import _get_folder
    assert "faiss_index" in _get_folder()

def test_hybrid_search(mocker, mock_embeddings, mock_cross_encoder, mock_faiss_db):
    # test hybrid search logic and ensemble retriever combination
    mocker.patch("os.path.isdir", return_value=True)
    
    # mock doc loading
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "mock content 1"
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "mock content 2"
    mocker.patch("pickle.load", return_value=[mock_doc1, mock_doc2])
    
    # mock retrievers
    mock_faiss_retriever = mock_faiss_db.as_retriever.return_value
    
    mock_bm25 = mocker.patch("hybrid_search._bm25_retriever")
    mock_bm25_retriever = mock_bm25
    
    # mock ensemble
    mock_ensemble = mocker.patch("hybrid_search.EnsembleRetriever")
    mock_ensemble_instance = mock_ensemble.return_value
    mock_ensemble_instance.get_relevant_documents.return_value = [mock_doc1, mock_doc2]
    
    # mock cross encoder scoring
    mock_ce = mocker.patch("hybrid_search._cross_encoder")
    mock_ce.predict.return_value = [0.1, 0.9]
    
    from hybrid_search import hybrid_search
    result = hybrid_search("test query")
    
    # check ensemble is configured correctly
    mock_ensemble.assert_called_once()
    kwargs = mock_ensemble.call_args[1]
    assert "retrievers" in kwargs
    assert kwargs["retrievers"] == [mock_faiss_retriever, mock_bm25_retriever]
    assert "weights" in kwargs
    assert kwargs["weights"] == [0.5, 0.5]
    
    # check cross encoder usage
    mock_ce.predict.assert_called_once()
    
    # check result is sorted by score and returns highest scoring doc first
    assert len(result) == 2
    assert result[0] == mock_doc2
