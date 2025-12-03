"""
Document ingestion and processing pipeline
Handles PDF, DOCX, TXT, and Markdown files
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import time

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.utils.vector_store import VectorStoreManager
from src.utils.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document ingestion, chunking, and indexing"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_manager: Optional[VectorStoreManager] = None,
        batch_size: int = 10  # NEW: Process in batches
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize text splitter with semantic separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " ", ""],
            keep_separator=True
        )
        
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.embedding_model = get_embedding_model()
        
    def load_document(self, file_path: str, file_type: Optional[str] = None) -> List[Document]:
        """Load document based on file type"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file type if not provided
        if file_type is None:
            file_type = path.suffix.lower()
        
        logger.info(f"Loading document: {file_path} (type: {file_type})")
        
        try:
            if file_type in ['.pdf']:
                loader = PyPDFLoader(file_path)
            elif file_type in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_type in ['.txt']:
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type in ['.md', '.markdown']:
                # For markdown, use TextLoader as fallback
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages/sections from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def chunk_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Split documents into chunks with metadata"""
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        
        # Add chunk-specific metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['chunk_index'] = idx
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_size'] = len(chunk.page_content)
            chunk.metadata['processed_at'] = datetime.utcnow().isoformat()
            
            # Calculate content hash for deduplication
            content_hash = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()
            chunk.metadata['content_hash'] = content_hash
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_type: Optional[str] = None,
        progress_callback = None  # NEW: For progress updates
    ) -> Dict[str, Any]:
        """Complete ingestion pipeline: load -> chunk -> embed -> index"""
        logger.info(f"Starting document ingestion: {file_path}")
        
        try:
            # Load document
            if progress_callback:
                progress_callback("Loading document...", 0.1)
            
            documents = self.load_document(file_path, file_type)
            
            # Add source metadata
            source_metadata = {
                "source": str(Path(file_path).name),
                "file_path": file_path,
                "file_type": file_type or Path(file_path).suffix,
                "ingested_at": datetime.utcnow().isoformat()
            }
            
            if metadata:
                source_metadata.update(metadata)
            
            # Chunk documents
            if progress_callback:
                progress_callback("Chunking document...", 0.3)
            
            chunks = self.chunk_documents(documents, source_metadata)
            
            # Index in batches to avoid rate limits
            if progress_callback:
                progress_callback(f"Indexing {len(chunks)} chunks...", 0.5)
            
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                self.vector_store_manager.add_documents(batch)
                
                if progress_callback:
                    progress = 0.5 + (0.5 * (batch_num / total_batches))
                    progress_callback(f"Batch {batch_num}/{total_batches}", progress)
                
                # Small delay to avoid rate limits
                if i + self.batch_size < len(chunks):
                    time.sleep(0.5)
            
            doc_id = source_metadata["source"]
            
            result = {
                "document_id": doc_id,
                "file_path": file_path,
                "num_pages": len(documents),
                "num_chunks": len(chunks),
                "metadata": source_metadata,
                "status": "success"
            }
            
            logger.info(f"Successfully ingested document: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            return {
                "file_path": file_path,
                "status": "error",
                "error": str(e)
            }
    
    def batch_ingest(
        self,
        file_paths: List[str],
        common_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Ingest multiple documents"""
        logger.info(f"Batch ingesting {len(file_paths)} documents")
        
        results = []
        for file_path in file_paths:
            result = self.ingest_document(file_path, common_metadata)
            results.append(result)
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Batch ingestion complete: {success_count}/{len(file_paths)} successful")
        
        return results
    
    def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for existing document chunks"""
        try:
            self.vector_store_manager.update_metadata(document_id, metadata)
            logger.info(f"Updated metadata for document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Remove document and its chunks from vector store"""
        try:
            self.vector_store_manager.delete_document(document_id)
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents"""
        return self.vector_store_manager.get_stats()
