from prometheus_client import Counter, Histogram, start_http_server
import time

# Define metrics
document_processing_time = Histogram(
    'document_processing_seconds',
    'Time spent processing documents'
)

documents_processed = Counter(
    'documents_processed_total',
    'Number of documents processed'
)

embedding_generation_time = Histogram(
    'embedding_generation_seconds',
    'Time spent generating embeddings'
)

vector_store_operations = Counter(
    'vector_store_operations_total',
    'Number of vector store operations',
    ['operation_type']
)