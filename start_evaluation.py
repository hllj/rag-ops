import logging
import json
from src.rag.chain import RAGChain
from src.ml.experiment import ExperimentManager

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize RAG chain
    rag_chain = RAGChain('config/config.yaml')
    experiment = ExperimentManager('config/config.yaml')
    
    # Load test questions from datasets/validation_set.json
    with open('datasets/validation_set.json', 'r') as f:
        validation_set = json.load(f)
        
    questions = [sample["questions"] for sample in validation_set]
    
    # Start MLFlow run
    with experiment.start_run(run_name="rag_evaluation"):
        # Log configuration
        experiment.log_params({
            "embedding_model": rag_chain.config['embedding_model']['model_name'],
            "llm_model": rag_chain.config['llm']['model_name'],
            "retriever_k": rag_chain.config['rag']['retriever']['k'],
            "retriever_type": rag_chain.config['rag']['retriever']['search_type'],
            "chunk_size": rag_chain.config['document_processor']['chunk_size']
        })
        
        # Run evaluation
        scores = rag_chain.evaluate(questions)
        
        # Log metrics
        experiment.log_metrics({
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"]
        })
        
        # Save validation dataset
        experiment.log_artifact("datasets/validation_set.json")
        
        # Log model configuration
        experiment.log_artifact("config/config.yaml")
        
        # Register model if metrics meet threshold
        if scores["answer_relevancy"] > 0.8 and scores["faithfulness"] > 0.8:
            model_version = experiment.register_model(
                model_name="rag-production-model"
            )
            
            # Transition to staging if metrics are good
            experiment.transition_model_stage(
                model_name="rag-production-model",
                version=model_version,
                stage="Staging"
            )

if __name__ == "__main__":
    main()