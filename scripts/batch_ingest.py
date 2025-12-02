"""
CompliAI - Batch ingest multiple policy documents
"""

import argparse
from pathlib import Path
from src.ingestion.document_processor import DocumentProcessor


def main():
    parser = argparse.ArgumentParser(description="CompliAI batch ingest policy documents")
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing policy documents"
    )
    parser.add_argument(
        "--department",
        type=str,
        default="General",
        help="Department for all documents"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="Policy",
        help="Policy type for all documents"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Get all files
    directory = Path(args.directory)
    files = list(directory.glob("**/*.pdf")) + \
            list(directory.glob("**/*.docx")) + \
            list(directory.glob("**/*.txt")) + \
            list(directory.glob("**/*.md"))
    
    print(f"CompliAI: Found {len(files)} documents to process")
    
    # Process each file
    metadata = {
        "department": args.department,
        "policy_type": args.policy_type
    }
    
    results = processor.batch_ingest(
        file_paths=[str(f) for f in files],
        common_metadata=metadata
    )
    
    # Print summary
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✅ CompliAI: Successfully processed: {success}/{len(files)}")
    
    for result in results:
        if result['status'] == 'success':
            print(f"  ✓ {result['file_path']}: {result['num_chunks']} chunks")
        else:
            print(f"  ✗ {result['file_path']}: {result['error']}")


if __name__ == "__main__":
    main()
