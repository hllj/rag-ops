import pytest
from unittest.mock import MagicMock, patch
from src.rag.chain import RAGChain

@pytest.fixture
def mock_retriever():
    return MagicMock()

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def rag_chain(test_config, mock_retriever, mock_llm):
    with patch('src.rag.chain.DocumentRetriever') as mock_doc_retriever, \
         patch('src.rag.chain.ChatOpenAI') as mock_chat:
        mock_doc_retriever.return_value = mock_retriever
        mock_chat.return_value = mock_llm
        chain = RAGChain(test_config)
        return chain

def test_rag_chain_initialization(rag_chain):
    assert rag_chain is not None
    assert hasattr(rag_chain, 'chain')
    assert hasattr(rag_chain, 'evaluator')

def test_rag_chain_query(rag_chain):
    test_question = "What is RAG?"
    mock_response = {
        "answer": "RAG is Retrieval-Augmented Generation",
        "source_documents": ["doc1", "doc2"]
    }
    rag_chain.chain = MagicMock(return_value=mock_response)
    
    result = rag_chain.query(test_question)
    
    assert result["answer"] == mock_response["answer"]
    assert result["source_documents"] == mock_response["source_documents"]

def test_rag_chain_evaluate(rag_chain):
    test_questions = ["What is RAG?", "How does RAG work?"]
    mock_scores = {
        "context_precision": 0.8,
        "context_recall": 0.7,
        "faithfulness": 0.9,
        "answer_relevancy": 0.85
    }
    
    rag_chain.evaluator.evaluate = MagicMock(return_value=mock_scores)
    
    scores = rag_chain.evaluate(test_questions)
    
    assert scores == mock_scores
    assert all(metric in scores for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"])
