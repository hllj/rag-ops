from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import yaml
import logging

class DocumentProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embedding_model = SentenceTransformer(
            self.config['embedding_model']['model_name']
        )
        self.chunk_size = self.config['document_processor']['chunk_size']
        self.chunk_overlap = self.config['document_processor']['chunk_overlap']
        
    def process_document(self, content: str) -> List[Dict]:
        """Process document content into chunks and generate embeddings."""
        chunks = self._create_chunks(content)
        embeddings = self._generate_embeddings(chunks)
        
        return [{
            'content': chunk,
            'embedding': embedding.tolist(),
            'metadata': {'chunk_id': i}
        } for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
    
    def _create_chunks(self, content: str) -> List[str]:
        """Split content into overlapping chunks."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def _generate_embeddings(self, chunks: List[str]):
        """Generate embeddings for text chunks."""
        return self.embedding_model.encode(chunks)