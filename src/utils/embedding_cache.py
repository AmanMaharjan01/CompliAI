"""
Local embedding cache to reduce API calls
"""

import logging
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache embeddings locally to avoid repeated API calls"""
    
    def __init__(self, cache_dir: str = "./data/embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized embedding cache: {cache_dir}")
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding"""
        cache_key = self._get_cache_key(text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return data['embedding']
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        
        return None
    
    def set(self, text: str, model: str, embedding: List[float]):
        """Cache embedding"""
        cache_key = self._get_cache_key(text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'text': text[:100],  # Store preview
                    'model': model,
                    'embedding': embedding
                }, f)
            logger.debug(f"Cached embedding for: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def clear(self):
        """Clear all cached embeddings"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cleared embedding cache")


# Global cache instance
_cache = EmbeddingCache()


def get_cached_embedding(text: str, model: str, embed_fn) -> List[float]:
    """
    Get embedding with caching
    
    Args:
        text: Text to embed
        model: Model name
        embed_fn: Function to call if not cached
    
    Returns:
        List of floats (embedding vector)
    """
    # Try cache first
    cached = _cache.get(text, model)
    if cached is not None:
        return cached
    
    # Call API
    embedding = embed_fn(text)
    
    # Cache result
    _cache.set(text, model, embedding)
    
    return embedding
