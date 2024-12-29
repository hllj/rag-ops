from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
import yaml
import logging

class DocumentRetriever:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']['model_name']
        )
        
        # Initialize vector store
        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.config['vector_store']['collection_name'],
            connection_args={
                "host": self.config['vector_store']['host'],
                "port": self.config['vector_store']['port']
            }
        )
        
        # Initialize text splitter for parent retriever
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['document_processor']['chunk_size'],
            chunk_overlap=self.config['document_processor']['chunk_overlap']
        )
        
        # Initialize retriever
        self.retriever = self._create_retriever()
        
    def _create_retriever(self):
        """Create the appropriate retriever based on configuration."""
        search_type = self.config['rag']['retriever']['search_type']
        
        if search_type == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config['rag']['retriever']['k'],
                    "fetch_k": self.config['rag']['retriever']['fetch_k'],
                    "lambda_mult": self.config['rag']['retriever']['lambda_mult']
                }
            )
        else:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config['rag']['retriever']['k']}
            )
            
    def retrieve(self, query: str):
        """Retrieve relevant documents for a query."""
        return self.retriever.get_relevant_documents(query)