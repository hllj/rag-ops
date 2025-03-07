import pytest
from unittest.mock import patch
from src.evaluation.rag_evaluation import RAGEvaluator

@pytest.fixture
def evaluator():
    return RAGEvaluator()

def test_evaluator_initialization(evaluator):
    assert evaluator is not None
    assert hasattr(evaluator, 'metrics')
    assert len(evaluator.metrics) == 4  # Should have 4 default metrics

@patch('src.evaluation.rag_evaluation.evaluate')
def test_evaluate(mock_evaluate, evaluator):
    questions = ["What is RAG?"]
    contexts = [["RAG is a framework."]]
    answers = ["RAG (Retrieval-Augmented Generation) is a framework."]
    
    mock_evaluate.return_value = {
        "context_precision": 0.8,
        "context_recall": 0.7,
        "faithfulness": 0.9,
        "answer_relevancy": 0.85
    }
    
    results = evaluator.evaluate(questions, contexts, answers)
    
    assert isinstance(results, dict)
    assert all(metric in results for metric in [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy"
    ])
    assert all(isinstance(score, float) for score in results.values())
    
    mock_evaluate.assert_called_once()

def test_evaluate_with_invalid_input(evaluator):
    with pytest.raises(Exception):
        evaluator.evaluate([], [], [])  # Empty inputs should raise an error
