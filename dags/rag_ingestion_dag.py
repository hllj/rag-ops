from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path properly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.document_ingestion_pipeline import DocumentIngestionPipeline
from src.pipeline.sources.source_folder import FolderSourceHandler

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_ingestion():
    config_path = os.path.join(PROJECT_ROOT, 'config/config.yaml')
    documents_path = os.path.join(PROJECT_ROOT, 'documents')
    
    pipeline = DocumentIngestionPipeline(config_path)
    folder_handler = FolderSourceHandler(pipeline.config, documents_path)
    pipeline.add_source_handler("folder", folder_handler)
    pipeline.run()

with DAG(
    'rag_ingestion',
    default_args=default_args,
    description='RAG document ingestion pipeline',
    schedule_interval=timedelta(hours=1),
    catchup=False
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_documents',
        python_callable=run_ingestion,
    )