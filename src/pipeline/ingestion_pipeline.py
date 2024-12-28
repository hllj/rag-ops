import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

from ..data.object_store import ObjectStore
from ..ml.experiment import ExperimentManager
from ..data.document_processor import DocumentProcessor
from ..data.vector_store import VectorStore
from ..monitoring.metrics import *

class DocumentIngestionPipeline(FileSystemEventHandler):
    def __init__(self, config_path: str, watch_directory: str):
        self.config_path = config_path
        self.watch_directory = watch_directory
        self.document_processor = DocumentProcessor(config_path)
        self.vector_store = VectorStore(config_path)
        self.object_store = ObjectStore(config_path)
        self.experiment_manager = ExperimentManager(config_path)
        self.logger = logging.getLogger(__name__)
        
    def on_created(self, event):
        """Handle new document creation."""
        if event.is_directory:
            return
            
        file_path = event.src_path
        if not self._is_supported_format(file_path):
            return
            
        self.logger.info(f"Processing new document: {file_path}")
        
        try:
            # Start MLflow run
            with self.experiment_manager.start_run(run_name=f"process_{os.path.basename(file_path)}"):
                # Log configuration parameters
                self.experiment_manager.log_params({
                    "chunk_size": self.document_processor.chunk_size,
                    "chunk_overlap": self.document_processor.chunk_overlap,
                    "model_name": self.document_processor.config['embedding_model']['model_name']
                })
                
                # Upload original document to MinIO
                object_name = self.object_store.upload_file(file_path)
                
                start_time = time.time()
                
                # Process document
                with document_processing_time.time():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    processed_chunks = self.document_processor.process_document(content)
                
                processing_time = time.time() - start_time
                
                # Store in vector database
                self.vector_store.upsert_documents(processed_chunks)
                
                # Log metrics
                self.experiment_manager.log_metrics({
                    "processing_time": processing_time,
                    "num_chunks": len(processed_chunks),
                    "avg_chunk_length": sum(len(chunk['content']) for chunk in processed_chunks) / len(processed_chunks)
                })
                
                # Update Prometheus metrics
                documents_processed.inc()
                vector_store_operations.labels(operation_type="insert").inc()
                
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")