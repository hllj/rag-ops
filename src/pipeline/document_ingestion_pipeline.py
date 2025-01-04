import yaml
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import os
import logging
from typing import Dict, Type

from .sources import SourceHandler, FolderWatchHandler
from ..data.object_store import ObjectStore
from ..ml.experiment import ExperimentManager
from ..data.document_processor import DocumentProcessor
from ..data.vector_store import VectorStore
from ..monitoring.metrics import *

class DocumentIngestionPipeline:
    """Main pipeline class for document ingestion from multiple sources."""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.document_processor = DocumentProcessor(config_path)
        self.vector_store = VectorStore(config_path)
        self.object_store = ObjectStore(config_path)
        self.experiment_manager = ExperimentManager(config_path)
        self.logger = logging.getLogger(__name__)

        # Initialize processing queue and thread pool
        self.processing_queue = queue.Queue()
        self.should_stop = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Initialize source handlers
        self.source_handlers: Dict[str, SourceHandler] = {}

    def add_source_handler(self, name: str, handler: SourceHandler) -> None:
        """Add a new source handler to the pipeline."""
        handler.set_processing_queue(self.processing_queue)
        self.source_handlers[name] = handler

    def run(self) -> None:
        """Start the document ingestion pipeline."""
        try:
            # Start all source handlers
            for handler in self.source_handlers.values():
                handler.start()

            # Start the processing worker
            processing_thread = threading.Thread(target=self._processing_worker)
            processing_thread.start()

            # Keep the main thread running
            while not self.should_stop.is_set():
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self.logger.info("Stopping document ingestion pipeline...")
        self.should_stop.set()
        
        # Stop all source handlers
        for handler in self.source_handlers.values():
            handler.stop()
            
        self.executor.shutdown(wait=True)
        self.logger.info("Pipeline stopped")

    def _processing_worker(self) -> None:
        """Worker thread to process files from the queue."""
        while not self.should_stop.is_set():
            try:
                # Get document from queue with timeout
                doc_info = self.processing_queue.get(timeout=1)
                self._process_document(doc_info)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing document: {str(e)}")

    def _process_document(self, doc_info: dict) -> None:
        """Process a single document."""
        file_path = doc_info["path"]
        try:
            # Start MLflow run
            with self.experiment_manager.start_run(run_name=f"process_{os.path.basename(file_path)}"):
                # Log configuration parameters
                self.experiment_manager.log_params({
                    "chunk_size": self.document_processor.chunk_size,
                    "chunk_overlap": self.document_processor.chunk_overlap,
                    "model_name": self.document_processor.config['embedding_model']['model_name'],
                    "source": doc_info["source"]
                })

                # Upload original document to MinIO
                object_name = self.object_store.upload_file(file_path)

                start_time = time.time()

                # Process document
                with document_processing_time.time():
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                        extra_info = {
                            "file_name": os.path.basename(file_path),
                            "source": doc_info["source"]
                        }
                        content = self.document_processor.parse_document(bytes=file_bytes, extra_info=extra_info)

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

                self.logger.info(f"Successfully processed file: {file_path}")

                # Handle processed file
                self._handle_processed_file(file_path)

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            self._handle_failed_file(file_path)

    def _handle_processed_file(self, file_path: str) -> None:
        """Handle a successfully processed file."""
        processed_dir = os.path.join(os.path.dirname(file_path), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        new_path = os.path.join(processed_dir, filename)
        os.rename(file_path, new_path)

    def _handle_failed_file(self, file_path: str) -> None:
        """Handle a file that failed processing."""
        error_dir = os.path.join(os.path.dirname(file_path), "error")
        os.makedirs(error_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        new_path = os.path.join(error_dir, filename)
        os.rename(file_path, new_path)