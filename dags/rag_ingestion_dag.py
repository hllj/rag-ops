from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    pipeline = DocumentIngestionPipeline('config/config.yaml')
    folder_handler = FolderSourceHandler(pipeline.config, "documents/")
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