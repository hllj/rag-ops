import pytest
import yaml
import os
import tempfile

@pytest.fixture
def test_config():
    config = {
        'embedding_model': {
            'model_name': 'all-MiniLM-L6-v2'
        },
        'vector_store': {
            'collection_name': 'test_collection'
        },
        'document_processor': {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'supported_formats': ['pdf', 'txt', 'docx']
        },
        'minio': {
            'access_key': 'test_access',
            'secret_key': 'test_secret',
            'bucket_name': 'test-bucket',
            'secure': False
        },
        'llm': {
            'provider': 'openai',
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 500
        },
        'rag': {
            'retriever': {
                'search_type': 'similarity',
                'k': 3
            },
            'chain': {
                'verbose': True
            }
        },
        'mlflow': {
            'experiment_name': 'test_experiment'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    yield config_path
    os.unlink(config_path)

@pytest.fixture
def sample_document():
    return "This is a test document content. It contains multiple sentences for testing purposes."

@pytest.fixture
def mocked_env(monkeypatch):
    monkeypatch.setenv("MINIO_ADDRESS", "localhost:9000")
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLAMA_PARSE_API", "test-key")
