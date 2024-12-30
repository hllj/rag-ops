import logging
from src.pipeline.ingestion_pipeline import DocumentIngestionPipeline

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run pipeline
    pipeline = DocumentIngestionPipeline(
        config_path='config/config.yaml',
        watch_directory='documents/'
    )
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()