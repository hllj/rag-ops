import logging
from src.pipeline.ingestion_pipeline import DocumentIngestionPipeline
from src.rag.chain import RAGChain
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
    
    # Start ingestion pipeline
    pipeline = DocumentIngestionPipeline(
        config_path='config/config.yaml',
        watch_directory='data/input'
    )
    
    # Start monitoring server
    start_http_server(config['monitoring']['metrics_port'])
    
    pipeline.run()
    
    # Initialize RAG chain
    rag_chain = RAGChain('config/config.yaml')
    
    # Example usage
    while True:
        try:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            response = rag_chain.query(question)
            print("\nAnswer:", response["answer"])
            print("\nSources:")
            for doc in response["source_documents"]:
                print(f"- {doc.metadata}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()