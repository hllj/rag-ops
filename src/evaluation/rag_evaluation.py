from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from typing import List, Dict
import logging

class RAGEvaluator:
    """Evaluates RAG pipeline performance using Ragas metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Define default metrics
        self.metrics = [
            context_precision,
            context_recall, 
            faithfulness,
            answer_relevancy
        ]

    def evaluate(self, questions: List[str], contexts: List[List[str]], answers: List[str]) -> Dict:
        """
        Evaluate RAG performance using Ragas metrics.
        
        Args:
            questions: List of questions
            contexts: List of context lists for each question
            answers: List of generated answers
            
        Returns:
            Dict containing evaluation scores
        """
        try:
            # Format data for Ragas
            dataset = {
                "question": questions,
                "contexts": contexts,
                "answer": answers
            }
            
            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics
            )
            
            # Convert results to dict
            scores = {
                "context_precision": float(results["context_precision"]),
                "context_recall": float(results["context_recall"]),
                "faithfulness": float(results["faithfulness"]),
                "answer_relevancy": float(results["answer_relevancy"])
            }
            
            self.logger.info(f"Evaluation results: {scores}")
            return scores
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise