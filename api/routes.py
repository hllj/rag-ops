from fastapi import APIRouter, Depends, HTTPException
from .models import QueryRequest, QueryResponse, HealthResponse
from .dependencies import get_rag_chain
from src.rag.chain import RAGChain
import logging
from typing import Dict

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
) -> Dict:
    try:
        response = rag_chain.query(
            question=request.question,
            chat_history=request.chat_history
        )
        
        return {
            "answer": response["answer"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response["source_documents"]
            ]
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict:
    return {
        "status": "healthy",
        "version": "1.0.0"
    }