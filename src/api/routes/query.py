"""
Query endpoints for policy questions
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.middleware.auth import get_current_user
from src.rag.query_engine import QueryEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize query engine (singleton)
query_engine = QueryEngine(
    use_reranking=True,
    use_multi_query=False,
    check_hallucinations=True
)


class QueryRequest(BaseModel):
    """Request model for policy query"""
    question: str = Field(..., min_length=5, max_length=500)
    filters: Optional[Dict[str, Any]] = Field(default=None)
    chat_history: Optional[str] = Field(default=None)
    k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response model for policy query"""
    question: str
    answer: Dict[str, Any]
    metadata: Dict[str, Any]
    status: str


@router.post("/query", response_model=QueryResponse)
async def query_policy(
    request: QueryRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Query policy documents with natural language question
    
    - **question**: Natural language question about company policy
    - **filters**: Optional filters (department, policy_type, etc.)
    - **chat_history**: Previous conversation context
    - **k**: Number of documents to retrieve (1-20)
    """
    try:
        logger.info(f"Query from user {current_user['user_id']}: {request.question}")
        
        result = query_engine.query(
            question=request.question,
            user_id=current_user['user_id'],
            filters=request.filters,
            chat_history=request.chat_history,
            k=request.k
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/history")
async def get_query_history(
    limit: int = Query(default=10, ge=1, le=100),
    current_user: Dict = Depends(get_current_user)
):
    """Get user's query history"""
    # Implement database query for history
    return {
        "user_id": current_user['user_id'],
        "queries": [],
        "total": 0
    }


@router.get("/query/suggestions")
async def get_query_suggestions(
    category: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Get suggested policy questions"""
    suggestions = {
        "HR": [
            "What is the PTO policy?",
            "How do I request parental leave?",
            "What are the remote work guidelines?"
        ],
        "IT": [
            "What is the password policy?",
            "How do I report a security incident?",
            "What software can I install?"
        ],
        "Legal": [
            "What is the code of conduct?",
            "How do I report ethical concerns?",
            "What is the confidentiality policy?"
        ]
    }
    
    if category:
        return {"category": category, "suggestions": suggestions.get(category, [])}
    
    return {"suggestions": suggestions}
