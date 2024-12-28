import yaml
from typing import List, Dict
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import logging

class VectorStore:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.collection_name = self.config['vector_store']['collection_name']
        self._connect()
        self._initialize_collection()
        
    def _connect(self):
        """Connect to vector database."""
        connections.connect(
            host=self.config['vector_store']['host'],
            port=self.config['vector_store']['port']
        )
        
    def _initialize_collection(self):
        """Initialize vector collection if it doesn't exist."""
        if self.collection_name not in Collection.list():
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            schema = CollectionSchema(fields=fields, description="Document chunks with embeddings")
            Collection(name=self.collection_name, schema=schema)
            
    def upsert_documents(self, documents: List[Dict]):
        """Insert or update documents in vector store."""
        collection = Collection(self.collection_name)
        
        # Prepare data in required format
        contents = [doc['content'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents]
        
        # Insert data
        collection.insert([contents, embeddings])
        collection.flush()
        
    def search(self, query_embedding: List[float], limit: int = 5):
        """Search for similar documents."""
        collection = Collection(self.collection_name)
        collection.load()
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["content"]
        )
        
        return results