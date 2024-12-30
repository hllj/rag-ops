import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

import yaml

from ..data.object_store import ObjectStore
from ..ml.experiment import ExperimentManager
from ..data.document_processor import DocumentProcessor
from ..data.vector_store import VectorStore
from ..monitoring.metrics import *
            

class DocumentIngestionPipeline(FileSystemEventHandler):
    def __init__(self, config_path: str, watch_directory: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.watch_directory = watch_directory
        self.document_processor = DocumentProcessor(config_path)
        self.vector_store = VectorStore(config_path)
        self.object_store = ObjectStore(config_path)
        self.experiment_manager = ExperimentManager(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing queue
        self.processing_queue = queue.Queue()
        self.should_stop = threading.Event()
        
        # Initialize thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def run(self):
        """Start the document ingestion pipeline."""
        try:
            # Start the file system observer
            observer = Observer()
            observer.schedule(self, self.watch_directory, recursive=False)
            observer.start()
            self.logger.info(f"Started watching directory: {self.watch_directory}")
            
            # Process any existing files in the directory
            self._process_existing_files()
            
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
        finally:
            observer.stop()
            observer.join()
            
    def stop(self):
        """Stop the pipeline gracefully."""
        self.logger.info("Stopping document ingestion pipeline...")
        self.should_stop.set()
        self.executor.shutdown(wait=True)
        self.logger.info("Pipeline stopped")
        
    def _process_existing_files(self):
        """Process any existing files in the watch directory."""
        for filename in os.listdir(self.watch_directory):
            file_path = os.path.join(self.watch_directory, filename)
            if os.path.isfile(file_path) and self._is_supported_format(file_path):
                self.logger.info(f"Found existing file: {filename}")
                self.processing_queue.put(file_path)
                
    def _processing_worker(self):
        """Worker thread to process files from the queue."""
        while not self.should_stop.is_set():
            try:
                # Get file from queue with timeout
                file_path = self.processing_queue.get(timeout=1)
                self._process_file(file_path)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing file: {str(e)}")
                
    def _process_file(self, file_path: str):
        """Process a single file."""
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
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                        extra_info = {
                            "file_name": os.path.basename(file_path),
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
                
                # Optionally move or delete the processed file
                self._handle_processed_file(file_path)
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            # Optionally move file to error directory
            self._handle_failed_file(file_path)
            
    def _handle_processed_file(self, file_path: str):
        """Handle a successfully processed file."""
        # Create processed directory if it doesn't exist
        processed_dir = os.path.join(self.watch_directory, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Move file to processed directory
        filename = os.path.basename(file_path)
        new_path = os.path.join(processed_dir, filename)
        os.rename(file_path, new_path)
        
    def _handle_failed_file(self, file_path: str):
        """Handle a file that failed processing."""
        # Create error directory if it doesn't exist
        error_dir = os.path.join(self.watch_directory, "error")
        os.makedirs(error_dir, exist_ok=True)
        
        # Move file to error directory
        filename = os.path.basename(file_path)
        new_path = os.path.join(error_dir, filename)
        os.rename(file_path, new_path)
        
    def on_created(self, event):
        """Handle file creation event."""
        if event.is_directory:
            return
            
        if self._is_supported_format(event.src_path):
            self.logger.info(f"New file detected: {event.src_path}")
            self.processing_queue.put(event.src_path)
            
    def _is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported."""
        extension = os.path.splitext(file_path)[1][1:].lower()
        return extension in self.config['document_processor']['supported_formats']