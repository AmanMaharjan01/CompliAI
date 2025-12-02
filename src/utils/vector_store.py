"""
CompliAI - Vector store management using ChromaDB
Handles document storage, retrieval, and similarity search
"""

import logging
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.schema import Document

from src.utils.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store operations for CompliAI"""
    
    def __init__(
        self,
        collection_name: str = "compliai_policies",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv(
            "VECTOR_STORE_DIR",
            "./data/vector_store"
        )
        
        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model()
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        
        # Initialize LangChain Chroma wrapper
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        logger.info(f"Initialized CompliAI vector store: {collection_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> str:
        """Add documents to vector store"""
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [doc.metadata.get('chunk_id') for doc in documents]
            
            # Add to vector store
            self.vector_store.add_documents(documents=documents, ids=ids)
            
            # Persist changes
            self.vector_store.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return documents[0].metadata.get('source', 'unknown')
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search"""
        try:
            logger.info(f"Similarity search: query='{query}', k={k}, filters={filter_dict}")
            
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with relevance scores"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} results with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """MMR search for diverse results"""
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_dict
            )
            
            logger.info(f"MMR search found {len(results)} diverse results")
            return results
            
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> None:
        """Delete document chunks by source"""
        try:
            # Delete by metadata filter
            collection = self.client.get_collection(self.collection_name)
            collection.delete(where={"source": document_id})
            
            self.vector_store.persist()
            logger.info(f"Deleted document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    def update_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Update metadata for document chunks"""
        try:
            collection = self.client.get_collection(self.collection_name)
            
            # Get all chunks for this document
            results = collection.get(where={"source": document_id})
            
            # Update each chunk's metadata
            for chunk_id in results['ids']:
                collection.update(
                    ids=[chunk_id],
                    metadatas=[metadata]
                )
            
            self.vector_store.persist()
            logger.info(f"Updated metadata for document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> None:
        """Clear all documents from collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(self.collection_name)
            logger.warning(f"Cleared collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
