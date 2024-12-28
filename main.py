import logging
from src.pipeline.ingestion_pipeline import DocumentIngestionPipeline
from src.monitoring.metrics import start_http_server
import yaml

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Start metrics server
    start_http_server(config['monitoring']['metrics_port'])
    
    # Start ingestion pipeline
    pipeline = DocumentIngestionPipeline(
        config_path='config/config.yaml',
        watch_directory='documents/'
    )
    pipeline.start()

if __name__ == "__main__":
    main()