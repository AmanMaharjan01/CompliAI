"""
Admin endpoints for document management
"""

import logging
from typing import List, Optional
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.api.middleware.auth import get_current_user, require_role
from src.ingestion.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize document processor
doc_processor = DocumentProcessor()


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    department: str
    policy_type: str
    effective_date: Optional[str] = None
    description: Optional[str] = None


class DocumentResponse(BaseModel):
    """Document upload response"""
    document_id: str
    filename: str
    status: str
    num_chunks: int
    message: str


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    policy_type: str = Form(...),
    effective_date: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user: Dict = Depends(require_role("admin"))
):
    """
    Upload policy document for ingestion
    
    Requires admin role
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file
        upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{current_user['user_id']}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved file: {file_path}")
        
        # Prepare metadata
        metadata = {
            "department": department,
            "policy_type": policy_type,
            "effective_date": effective_date,
            "description": description,
            "uploaded_by": current_user['user_id']
        }
        
        # Ingest document
        result = doc_processor.ingest_document(
            file_path=str(file_path),
            metadata=metadata,
            file_type=file_ext
        )
        
        if result['status'] == 'success':
            return DocumentResponse(
                document_id=result['document_id'],
                filename=file.filename,
                status="success",
                num_chunks=result['num_chunks'],
                message=f"Document successfully ingested with {result['num_chunks']} chunks"
            )
        else:
            raise HTTPException(status_code=500, detail=result.get('error'))
            
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    department: Optional[str] = None,
    policy_type: Optional[str] = None,
    current_user: Dict = Depends(require_role("admin"))
):
    """List all ingested documents"""
    # Implement database query
    return {
        "documents": [],
        "total": 0,
        "filters": {
            "department": department,
            "policy_type": policy_type
        }
    }


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Dict = Depends(require_role("admin"))
):
    """Delete document and its chunks"""
    try:
        success = doc_processor.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def reindex_documents(
    current_user: Dict = Depends(require_role("admin"))
):
    """Reindex all documents"""
    return {"message": "Reindexing started", "status": "processing"}


@router.get("/analytics")
async def get_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: Dict = Depends(require_role("admin"))
):
    """Get analytics dashboard data"""
    return {
        "total_queries": 0,
        "avg_response_time": 0,
        "top_questions": [],
        "department_breakdown": {},
        "confidence_distribution": {}
    }
