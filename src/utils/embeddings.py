"""
Embedding model configuration and utilities
"""

import logging
import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_embedding_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> OpenAIEmbeddings:
    """Initialize and return embedding model"""
    
    model_name = model_name or os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small"
    )
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key
    )
    
    logger.info(f"Initialized embedding model: {model_name}")
    return embeddings
