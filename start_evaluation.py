import logging
from src.rag.chain import RAGChain
import json

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize RAG chain
    rag_chain = RAGChain('config/config.yaml')
    
    # Load test questions from datasets/validation_set.json
    with open('datasets/validation_set.json', 'r') as f:
        validation_set = json.load(f)
        
    questions = [sample["questions"] for sample in validation_set]
    
    rag_chain.evaluate(questions)

if __name__ == "__main__":
    main()