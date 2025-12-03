"""
Embedding model configuration with FREE local option
"""

import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)


class CachedEmbeddings:
    """Wrapper for embeddings with local caching"""
    
    def __init__(self, base_embeddings, model_name: str, use_cache: bool = True):
        self.base_embeddings = base_embeddings
        self.model_name = model_name
        self.use_cache = use_cache
        
        if use_cache:
            from src.utils.embedding_cache import EmbeddingCache
            self.cache = EmbeddingCache()
            logger.info("Embedding caching enabled - will reduce API calls")
        else:
            self.cache = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with caching"""
        if not self.use_cache:
            return self.base_embeddings.embed_documents(texts)
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch embed uncached texts
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)}/{len(texts)} uncached documents")
            new_embeddings = self.base_embeddings.embed_documents(uncached_texts)
            
            # Cache and insert new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                self.cache.set(uncached_texts[uncached_indices.index(idx)], 
                             self.model_name, 
                             embedding)
                embeddings[idx] = embedding
        else:
            logger.info(f"All {len(texts)} documents found in cache!")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query with caching"""
        if not self.use_cache:
            return self.base_embeddings.embed_query(text)
        
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            logger.debug("Query embedding found in cache")
            return cached
        
        logger.debug("Query embedding not in cache, calling API")
        embedding = self.base_embeddings.embed_query(text)
        self.cache.set(text, self.model_name, embedding)
        
        return embedding


def get_embedding_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True
):
    """Initialize embedding model based on EMBEDDING_PROVIDER from .env"""
    
    # Read from environment
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    use_cache_env = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
    use_cache = use_cache and use_cache_env
    
    logger.info(f"Embedding Configuration from .env:")
    logger.info(f"  - Provider: {provider}")
    logger.info(f"  - Caching: {use_cache}")
    
    try:
        if provider == "huggingface":
            return get_huggingface_embeddings(model_name, use_cache)
        elif provider == "gemini":
            return get_gemini_embedding_model(model_name, api_key, use_cache)
        elif provider == "openai":
            return get_openai_embeddings(model_name, api_key, use_cache)
        else:
            logger.warning(f"Unknown provider '{provider}', defaulting to HuggingFace (free)")
            return get_huggingface_embeddings(model_name, use_cache)
    except Exception as e:
        logger.error(f"Failed to initialize {provider} embeddings: {str(e)}")
        
        # Try fallback providers
        if provider != "huggingface":
            try:
                logger.info("Attempting fallback to HuggingFace...")
                return get_huggingface_embeddings(model_name, use_cache)
            except Exception as e2:
                logger.error(f"HuggingFace fallback also failed: {str(e2)}")
        
        raise ValueError(
            f"Failed to initialize embeddings with provider '{provider}'.\n"
            f"Error: {str(e)}\n\n"
            f"To fix:\n"
            f"1. For HuggingFace: pip3 install sentence-transformers\n"
            f"2. For Gemini: Set GOOGLE_API_KEY in .env\n"
            f"3. For OpenAI: Set OPENAI_API_KEY in .env"
        )


def get_huggingface_embeddings(
    model_name: Optional[str] = None,
    use_cache: bool = True
):
    """FREE local embeddings using HuggingFace - reads from .env"""
    
    # Read from environment
    model_name = model_name or os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info(f"HuggingFace model from .env: {model_name}")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        logger.info(f"Loading FREE local embedding model: {model_name}")
        logger.info("⚡ This runs on your computer - NO API calls!")
        logger.info("📥 First time may take 1-2 minutes to download model...")
        
        base_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU (works on any computer)
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if use_cache:
            embeddings = CachedEmbeddings(base_embeddings, model_name, use_cache)
        else:
            embeddings = base_embeddings
        
        logger.info("✅ HuggingFace embeddings initialized (FREE, NO API CALLS)")
        return embeddings
        
    except ImportError as e:
        logger.error("sentence-transformers package not installed")
        raise ValueError(
            "HuggingFace embeddings require sentence-transformers.\n"
            "Install with: pip3 install sentence-transformers\n\n"
            f"Original error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {str(e)}")
        raise ValueError(f"Failed to initialize HuggingFace embeddings: {str(e)}")


def get_openai_embeddings(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True
):
    """OpenAI embeddings - reads from .env"""
    
    # Read from environment
    model_name = model_name or os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small"
    )
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    logger.info(f"OpenAI Configuration from .env:")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - API Key: {api_key[:8] if api_key else 'Not set'}...")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        from langchain_openai import OpenAIEmbeddings
        
        base_embeddings = OpenAIEmbeddings()
        
        if use_cache:
            embeddings = CachedEmbeddings(base_embeddings, model_name, use_cache)
        else:
            embeddings = base_embeddings
        
        logger.info(f"Initialized OpenAI embeddings (API calls required)")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise ValueError(f"Failed to initialize OpenAI embeddings: {str(e)}")


def get_gemini_embedding_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True
):
    """Google Gemini embeddings - reads from .env"""
    
    # Read from environment
    model_name = model_name or os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "models/embedding-001"
    )
    
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    
    logger.info(f"Gemini Embedding Configuration from .env:")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - API Key: {api_key[:8] if api_key else 'Not set'}...")
    
    if not api_key:
        raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file")
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )
        
        if use_cache:
            embeddings = CachedEmbeddings(base_embeddings, model_name, use_cache)
        else:
            embeddings = base_embeddings
        
        logger.info(f"Initialized Gemini embeddings (API calls with limits)")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize Gemini embeddings: {str(e)}")
        raise ValueError(f"Failed to initialize Gemini embeddings: {str(e)}")
