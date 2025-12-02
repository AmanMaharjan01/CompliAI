"""
CompliAI - Script to run RAG evaluation
"""

import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate CompliAI RAG pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compliai_evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation(
        test_file=args.dataset,
        output_file=args.output
    )
    
    # Print summary
    print("\n=== CompliAI Evaluation Results ===")
    print(f"\nRetrieval Metrics:")
    for metric, value in results["retrieval_metrics"].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nAnswer Metrics:")
    for metric, value in results["answer_metrics"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
