from pydantic import BaseModel
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

class SourceDocument(BaseModel):
    content: str
    metadata: Dict

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
    chat_history: Optional[List[Dict[str, str]]]

class HealthResponse(BaseModel):
    status: str
    version: str