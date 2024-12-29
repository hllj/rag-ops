from functools import lru_cache
from src.rag.chain import RAGChain

@lru_cache()
def get_rag_chain():
    """Singleton pattern for RAG chain initialization."""
    return RAGChain('config/config.yaml')