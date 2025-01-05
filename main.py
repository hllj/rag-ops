import logging
from src.rag.chain import RAGChain
from src.monitoring.metrics import start_http_server
import yaml

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
                print(f"- {doc}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()