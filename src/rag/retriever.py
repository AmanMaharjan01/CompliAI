"""
Advanced retrieval strategies including hybrid search and reranking
"""

import logging
from typing import List, Dict, Any, Optional
import os

from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import cohere

from src.utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines semantic and keyword search with reranking"""
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        use_reranking: bool = True,
        use_compression: bool = False
    ):
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.use_reranking = use_reranking
        self.use_compression = use_compression
        
        # Initialize reranker
        if use_reranking:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                self.reranker = cohere.Client(cohere_api_key)
                logger.info("Initialized Cohere reranker")
            else:
                logger.warning("Cohere API key not found, reranking disabled")
                self.use_reranking = False
        
        # Initialize compression
        if use_compression:
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            self.compressor = LLMChainExtractor.from_llm(llm)
            logger.info("Initialized contextual compression")
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        rerank_top_n: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """Main retrieval method with hybrid search and reranking"""
        
        logger.info(f"Retrieving documents for query: '{query[:100]}...'")
        
        # Step 1: Semantic search
        semantic_results = self.vector_store_manager.similarity_search_with_score(
            query=query,
            k=k,
            filter_dict=filter_dict
        )
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in semantic_results
            if score >= score_threshold
        ]
        
        logger.info(f"Semantic search returned {len(filtered_results)} documents")
        
        if not filtered_results:
            logger.warning("No documents found above score threshold")
            return []
        
        # Step 2: Reranking (optional)
        if self.use_reranking and len(filtered_results) > rerank_top_n:
            documents = self._rerank_documents(
                query=query,
                documents=[doc for doc, _ in filtered_results],
                top_n=rerank_top_n
            )
        else:
            documents = [doc for doc, _ in filtered_results[:rerank_top_n]]
        
        # Step 3: Contextual compression (optional)
        if self.use_compression:
            documents = self._compress_documents(query, documents)
        
        logger.info(f"Final retrieval: {len(documents)} documents")
        return documents
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5
    ) -> List[Document]:
        """Rerank documents using Cohere"""
        try:
            # Prepare documents for reranking
            doc_texts = [doc.page_content for doc in documents]
            
            # Call Cohere rerank API
            results = self.reranker.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_n,
                model="rerank-english-v2.0"
            )
            
            # Reorder documents based on rerank scores
            reranked_docs = []
            for result in results.results:
                doc = documents[result.index]
                doc.metadata['rerank_score'] = result.relevance_score
                reranked_docs.append(doc)
            
            logger.info(f"Reranked to top {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}, using original order")
            return documents[:top_n]
    
    def _compress_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """Extract only relevant sentences from documents"""
        try:
            compressed = self.compressor.compress_documents(
                documents=documents,
                query=query
            )
            logger.info(f"Compressed {len(documents)} documents")
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {str(e)}")
            return documents
    
    def retrieve_with_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve diverse documents using MMR"""
        
        documents = self.vector_store_manager.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter_dict=filter_dict
        )
        
        logger.info(f"MMR retrieval: {len(documents)} diverse documents")
        return documents


class MultiQueryRetriever:
    """Generate multiple query variations for comprehensive retrieval"""
    
    def __init__(self, base_retriever: HybridRetriever):
        self.base_retriever = base_retriever
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    
    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate alternative phrasings of the query"""
        prompt = f"""Generate {num_variations} alternative phrasings of this question
that capture the same intent but use different wording:

Original: {query}

Variations (one per line):"""
        
        try:
            response = self.llm.predict(prompt)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return [query] + variations[:num_variations]
        except Exception as e:
            logger.error(f"Query variation generation failed: {str(e)}")
            return [query]
    
    def retrieve(
        self,
        query: str,
        k_per_query: int = 5,
        **kwargs
    ) -> List[Document]:
        """Retrieve using multiple query variations and deduplicate"""
        
        # Generate variations
        queries = self.generate_query_variations(query)
        logger.info(f"Generated {len(queries)} query variations")
        
        # Retrieve for each variation
        all_docs = []
        seen_content = set()
        
        for q in queries:
            docs = self.base_retriever.retrieve(query=q, k=k_per_query, **kwargs)
            
            # Deduplicate by content hash
            for doc in docs:
                content_hash = doc.metadata.get('content_hash')
                if content_hash and content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        logger.info(f"Multi-query retrieval: {len(all_docs)} unique documents")
        return all_docs
