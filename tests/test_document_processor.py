import pytest
from unittest.mock import MagicMock, patch
from src.data.document_processor import DocumentProcessor

@pytest.fixture
def doc_processor(test_config):
    with patch('src.data.document_processor.SentenceTransformer') as mock_transformer, \
         patch('src.data.document_processor.LlamaParse') as mock_parser:
        processor = DocumentProcessor(test_config)
        return processor

def test_document_processor_initialization(doc_processor):
    assert doc_processor is not None
    assert hasattr(doc_processor, 'chunk_size')
    assert hasattr(doc_processor, 'chunk_overlap')

def test_create_chunks(doc_processor, sample_document):
    chunks = doc_processor._create_chunks(sample_document)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_generate_embeddings(doc_processor):
    test_chunks = ["chunk1", "chunk2"]
    mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    doc_processor.embedding_model.encode = MagicMock(return_value=mock_embeddings)
    
    embeddings = doc_processor._generate_embeddings(test_chunks)
    assert len(embeddings) == len(test_chunks)
    doc_processor.embedding_model.encode.assert_called_once_with(test_chunks)

def test_process_document(doc_processor, sample_document):
    doc_processor._create_chunks = MagicMock(return_value=["chunk1", "chunk2"])
    doc_processor._generate_embeddings = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    
    result = doc_processor.process_document(sample_document)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, dict) for item in result)
    assert all(key in result[0] for key in ['content', 'embedding', 'metadata'])
