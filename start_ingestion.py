import logging
from src.pipeline.document_ingestion_pipeline import DocumentIngestionPipeline
from src.pipeline.sources.source_folder import FolderSourceHandler

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run pipeline
    pipeline = DocumentIngestionPipeline(
        config_path='config/config.yaml'
    )
    
    # Add a folder source handler
    folder_handler = FolderSourceHandler(pipeline.config, "documents/")
    pipeline.add_source_handler("folder", folder_handler)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()