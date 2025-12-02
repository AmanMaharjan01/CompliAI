"""
Main RAG query engine orchestrating retrieval and generation
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from langchain.schema import Document

from src.rag.retriever import HybridRetriever, MultiQueryRetriever
from src.rag.generator import AnswerGenerator, GenerationResult
from src.utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class QueryEngine:
    """Main RAG pipeline orchestrator"""
    
    def __init__(
        self,
        use_reranking: bool = True,
        use_multi_query: bool = False,
        check_hallucinations: bool = True
    ):
        # Initialize components
        self.vector_store_manager = VectorStoreManager()
        
        base_retriever = HybridRetriever(
            vector_store_manager=self.vector_store_manager,
            use_reranking=use_reranking
        )
        
        if use_multi_query:
            self.retriever = MultiQueryRetriever(base_retriever)
        else:
            self.retriever = base_retriever
        
        self.generator = AnswerGenerator(
            check_hallucinations=check_hallucinations
        )
        
        logger.info("Initialized QueryEngine")
    
    def query(
        self,
        question: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        chat_history: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """Execute complete RAG query pipeline"""
        
        start_time = time.time()
        logger.info(f"Processing query from user {user_id}: '{question[:100]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(
                query=question,
                k=k,
                filter_dict=filters
            )
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            if not retrieved_docs:
                return self._create_no_context_response(question)
            
            # Step 2: Generate answer
            generation_start = time.time()
            result = self.generator.generate(
                question=question,
                retrieved_docs=retrieved_docs,
                chat_history=chat_history
            )
            generation_time = (time.time() - generation_start) * 1000
            
            # Step 3: Format response
            total_time = (time.time() - start_time) * 1000
            
            response = self._format_response(
                result=result,
                question=question,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                user_id=user_id
            )
            
            logger.info(
                f"Query completed in {total_time:.2f}ms "
                f"(retrieval: {retrieval_time:.2f}ms, generation: {generation_time:.2f}ms)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return self._create_error_response(question, str(e))
    
    def _format_response(
        self,
        result: GenerationResult,
        question: str,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Format complete query response"""
        
        answer = result.answer
        
        return {
            "question": question,
            "answer": {
                "summary": answer.summary,
                "detailed_answer": answer.detailed_answer,
                "policy_references": answer.policy_references,
                "confidence": {
                    "level": answer.confidence_level,
                    "reasoning": answer.confidence_reasoning
                },
                "action_items": answer.action_items or [],
                "related_topics": answer.related_topics
            },
            "metadata": {
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "num_sources": len(result.retrieved_docs),
                "sources": [
                    {
                        "document": doc.metadata.get('source'),
                        "page": doc.metadata.get('page'),
                        "chunk_id": doc.metadata.get('chunk_id')
                    }
                    for doc in result.retrieved_docs
                ],
                "performance": {
                    "retrieval_time_ms": round(retrieval_time, 2),
                    "generation_time_ms": round(generation_time, 2),
                    "total_time_ms": round(total_time, 2)
                },
                "quality": {
                    "is_grounded": result.is_grounded,
                    "hallucination_score": result.hallucination_score,
                    "requires_escalation": answer.requires_escalation
                }
            },
            "status": "success"
        }
    
    def _create_no_context_response(self, question: str) -> Dict[str, Any]:
        """Response when no relevant documents found"""
        return {
            "question": question,
            "answer": {
                "summary": "I don't have enough information to answer this question.",
                "detailed_answer": (
                    "I couldn't find relevant policy documents to answer your question. "
                    "This might be because:\n"
                    "1. The policy hasn't been uploaded to the system yet\n"
                    "2. Your question requires information from a different department\n"
                    "3. The policy might use different terminology\n\n"
                    "Please try:\n"
                    "- Rephrasing your question\n"
                    "- Contacting HR or the relevant department directly\n"
                    "- Checking if the policy exists in your department's resources"
                ),
                "policy_references": [],
                "confidence": {
                    "level": "N/A",
                    "reasoning": "No relevant documents retrieved"
                },
                "action_items": [
                    "Contact HR for clarification",
                    "Check department-specific resources"
                ],
                "related_topics": []
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "num_sources": 0,
                "sources": []
            },
            "status": "no_context"
        }
    
    def _create_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Response for system errors"""
        return {
            "question": question,
            "answer": {
                "summary": "An error occurred while processing your question.",
                "detailed_answer": (
                    "I encountered a technical issue while trying to answer your question. "
                    "Please try again in a moment. If the issue persists, contact technical support."
                ),
                "policy_references": [],
                "confidence": {"level": "N/A", "reasoning": "System error"},
                "action_items": ["Try again", "Contact support if issue persists"],
                "related_topics": []
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "error": error
            },
            "status": "error"
        }
