import pytest
from unittest.mock import patch, MagicMock
from src.data.vector_store import VectorStore

@pytest.fixture
def vector_store(test_config, mocked_env):
    with patch('src.data.vector_store.connections'), \
         patch('src.data.vector_store.Collection'), \
         patch('src.data.vector_store.utility'):
        store = VectorStore(test_config)
        return store

def test_vector_store_initialization(vector_store):
    assert vector_store is not None
    assert hasattr(vector_store, 'collection_name')

@patch('src.data.vector_store.Collection')
def test_upsert_documents(mock_collection, vector_store):
    mock_coll_instance = MagicMock()
    mock_collection.return_value = mock_coll_instance
    
    test_docs = [
        {
            'content': 'test content',
            'embedding': [0.1, 0.2, 0.3]
        }
    ]
    
    vector_store.upsert_documents(test_docs)
    
    mock_coll_instance.insert.assert_called_once()
    mock_coll_instance.flush.assert_called_once()
    mock_coll_instance.load.assert_called_once()

@patch('src.data.vector_store.Collection')
def test_search(mock_collection, vector_store):
    mock_coll_instance = MagicMock()
    mock_collection.return_value = mock_coll_instance
    
    query_embedding = [0.1, 0.2, 0.3]
    vector_store.search(query_embedding, limit=5)
    
    mock_coll_instance.load.assert_called_once()
    mock_coll_instance.search.assert_called_once()

def test_get_collection_stats(vector_store):
    with patch('src.data.vector_store.utility.has_collection', return_value=True), \
         patch('src.data.vector_store.Collection') as mock_collection:
        mock_coll_instance = MagicMock()
        mock_coll_instance.stats.return_value = {"row_count": 100}
        mock_collection.return_value = mock_coll_instance
        
        stats = vector_store.get_collection_stats()
        
        assert stats["row_count"] == 100
        assert stats["collection_name"] == vector_store.collection_name
