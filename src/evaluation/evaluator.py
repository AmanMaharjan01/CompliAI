"""
CompliAI - RAG evaluation pipeline using LangSmith
"""

import logging
from typing import List, Dict, Any
import json
from pathlib import Path

from langsmith import Client
from langsmith.evaluation import evaluate

from src.rag.query_engine import QueryEngine

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate CompliAI RAG pipeline performance"""
    
    def __init__(self, langsmith_client: Client = None):
        self.client = langsmith_client or Client()
        self.query_engine = QueryEngine()
    
    def create_test_dataset(
        self,
        test_cases: List[Dict[str, Any]],
        dataset_name: str = "compliai_qa_test"
    ) -> str:
        """Create LangSmith dataset from test cases"""
        
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description="CompliAI Policy Q&A test cases"
        )
        
        for case in test_cases:
            self.client.create_example(
                inputs={"question": case["question"]},
                outputs={"expected_answer": case["expected_answer"]},
                dataset_id=dataset.id,
                metadata=case.get("metadata", {})
            )
        
        logger.info(f"Created CompliAI dataset {dataset_name} with {len(test_cases)} examples")
        return dataset.id
    
    def evaluate_retrieval_quality(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate retrieval accuracy"""
        
        metrics = {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": []  # Mean Reciprocal Rank
        }
        
        for case in test_cases:
            question = case["question"]
            expected_doc_ids = set(case.get("relevant_doc_ids", []))
            
            # Retrieve documents
            result = self.query_engine.query(question=question, k=10)
            
            retrieved_doc_ids = set([
                source["chunk_id"]
                for source in result["metadata"]["sources"]
            ])
            
            # Calculate metrics
            if expected_doc_ids:
                # Precision@K
                precision = len(expected_doc_ids & retrieved_doc_ids) / len(retrieved_doc_ids) if retrieved_doc_ids else 0
                metrics["precision_at_k"].append(precision)
                
                # Recall@K
                recall = len(expected_doc_ids & retrieved_doc_ids) / len(expected_doc_ids)
                metrics["recall_at_k"].append(recall)
                
                # MRR
                for i, doc_id in enumerate(retrieved_doc_ids, 1):
                    if doc_id in expected_doc_ids:
                        metrics["mrr"].append(1 / i)
                        break
                else:
                    metrics["mrr"].append(0)
        
        # Calculate averages
        return {
            metric: sum(values) / len(values) if values else 0
            for metric, values in metrics.items()
        }
    
    def evaluate_answer_quality(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate answer accuracy and relevance"""
        
        results = {
            "total": len(test_cases),
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "high_confidence": 0,
            "avg_response_time": []
        }
        
        for case in test_cases:
            result = self.query_engine.query(question=case["question"])
            
            # Manual evaluation required or use LLM-as-judge
            # For now, check confidence levels
            confidence = result["answer"]["confidence"]["level"]
            
            if confidence == "High":
                results["high_confidence"] += 1
            
            results["avg_response_time"].append(
                result["metadata"]["performance"]["total_time_ms"]
            )
        
        results["avg_response_time"] = sum(results["avg_response_time"]) / len(results["avg_response_time"])
        
        return results
    
    def run_evaluation(
        self,
        test_file: str,
        output_file: str = "evaluation_results.json"
    ):
        """Run complete evaluation pipeline"""
        
        logger.info(f"Loading test cases from {test_file}")
        
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
        
        logger.info(f"Evaluating {len(test_cases)} test cases")
        
        # Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval_quality(test_cases)
        
        # Evaluate answers
        answer_metrics = self.evaluate_answer_quality(test_cases)
        
        # Combine results
        results = {
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "test_cases_count": len(test_cases)
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
        
        return results
