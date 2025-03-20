from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.chain import RAGChain
from src.ml.experiment import ExperimentManager
import json

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def evaluate_rag():
    # Initialize components
    rag_chain = RAGChain('config/config.yaml')
    experiment = ExperimentManager('config/config.yaml')
    
    # Load validation data
    with open('datasets/validation_set.json', 'r') as f:
        validation_set = json.load(f)
    questions = [sample["question"] for sample in validation_set]
    
    # Run evaluation
    with experiment.start_run(run_name=f"automated_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}"):
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
        experiment.log_metrics(scores)
        
        # Register model if metrics meet thresholds
        if (scores["answer_relevancy"] > rag_chain.config['mlflow']['model_registry']['metric_thresholds']['answer_relevancy'] and 
            scores["faithfulness"] > rag_chain.config['mlflow']['model_registry']['metric_thresholds']['faithfulness']):
            
            version = experiment.register_model(
                model_name=rag_chain.config['mlflow']['model_registry']['model_name']
            )
            
            # Transition to staging
            experiment.transition_model_stage(
                model_name=rag_chain.config['mlflow']['model_registry']['model_name'],
                version=version,
                stage="Staging"
            )

with DAG(
    'rag_evaluation',
    default_args=default_args,
    description='RAG evaluation pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    evaluate_task = PythonOperator(
        task_id='evaluate_rag',
        python_callable=evaluate_rag,
    )