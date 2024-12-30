import os
from typing import List, Dict
import yaml
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse


class DocumentProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embedding_model = SentenceTransformer(
            self.config['embedding_model']['model_name']
        )
        self.parser = LlamaParse(
            api_key=os.environ["LLAMA_PARSE_API"],
            result_type="markdown"
        )
        self.chunk_size = self.config['document_processor']['chunk_size']
        self.chunk_overlap = self.config['document_processor']['chunk_overlap']
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
    def parse_document(self, bytes: bytes, extra_info: dict) -> str:
        """Parse a document from bytes to text."""
        documents = self.parser.load_data(bytes, extra_info=extra_info)
        content = "".join([doc.text for doc in documents])
        return content
        
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
        texts = self.text_splitter.create_documents([content])
        return [text.page_content for text in texts]
    
    def _generate_embeddings(self, chunks: List[str]):
        """Generate embeddings for text chunks."""
        return self.embedding_model.encode(chunks)