from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.document_processor import DocumentProcessor
from app.models.document import Document
from app.core.config import settings
from app.api.auth import get_current_user  # Add this import
from app.models.user import User
from typing import Annotated
import os
import uuid
from openai import OpenAIError
import logging
from starlette.responses import FileResponse
import json
from sqlalchemy import or_, cast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import String

logger = logging.getLogger(__name__)
router = APIRouter()
document_processor = DocumentProcessor()

@router.post("/upload")
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Validate file extension
        ext = file.filename.split(".")[-1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="File type not supported")

        # Process file content
        content = ""
        if ext == "pdf":
            content = document_processor.process_pdf(file.file)
        elif ext == "docx":
            content = document_processor.process_docx(file.file)
        elif ext == "md":
            content = document_processor.process_markdown(await file.read())

        # Analyze document content
        analysis = await document_processor.analyze_document(content)

        # Save file
        file_path = f"{settings.DOCUMENT_STORAGE_PATH}/{uuid.uuid4()}.{ext}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            file.file.seek(0)
            f.write(await file.read())

        # Create document record
        document = Document(
            title=file.filename,
            content=content,
            file_path=file_path,
            file_type=ext,
            embedding=analysis["embedding"],
            metadata_col={
                "summary": analysis["summary"],
                "keywords": analysis["keywords"]
            },
            user_id=current_user.id
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("")
async def get_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    documents = db.query(Document).filter(
        Document.user_id == current_user.id
    ).order_by(Document.created_at.desc()).all()
    
    return documents 


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="Document file not found")
    
    headers = {
        "Content-Disposition": f'inline; filename="{document.title}"',
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true"
    }
    
    return FileResponse(
        path=document.file_path,
        media_type=f"application/{document.file_type}",
        filename=document.title,
        headers=headers
    )

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete the physical file
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        
    # Delete from database
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}

@router.get("/search")
async def search_documents(
    q: str,
    filters: str | None = None,
    search_type: str = "hybrid",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Parse filters if provided
    filter_dict = json.loads(filters) if filters else {}
    
    # Preprocess search query
    query_terms = q.lower().split()
    
    # Base query
    query = db.query(Document).filter(Document.user_id == current_user.id)
    
    # Apply metadata filters
    if filter_dict:
        for key, value in filter_dict.items():
            if value:
                query = query.filter(Document.metadata_col[key].contains(value))

    results = []
    scored_results = []
    
    if search_type in ["semantic", "hybrid"]:
        # Get all documents for scoring
        all_docs = query.all()
        query_embedding = document_processor.create_embeddings(q)
        
        # Score documents using embeddings
        for doc in all_docs:
            if doc.embedding:
                similarity = cosine_similarity(
                    [query_embedding],
                    [doc.embedding]
                )[0][0]
                scored_results.append((doc, float(similarity)))
    
    if search_type in ["keyword", "hybrid"]:
        # Get keyword matches
        keyword_matches = query.filter(
            or_(
                Document.title.ilike(f"%{q}%"),
                Document.content.ilike(f"%{q}%"),
                Document.metadata_col['keywords'].cast(String).ilike(f"%{q}%")
            )
        ).all()
        
        # Score keyword matches
        for doc in keyword_matches:
            score = 0
            if q.lower() in doc.title.lower():
                score += 0.8
            if q.lower() in doc.content.lower():
                score += 0.5
            if doc.metadata_col.get('keywords') and \
               any(q.lower() in k.lower() for k in doc.metadata_col['keywords']):
                score += 0.3
                
            scored_results.append((doc, score))
    
    # Remove duplicates and sort by score
    seen_ids = set()
    unique_results = []
    for doc, score in sorted(scored_results, key=lambda x: x[1], reverse=True):
        if doc.id not in seen_ids:
            seen_ids.add(doc.id)
            unique_results.append({
                "id": doc.id,
                "title": doc.title,
                "excerpt": doc.content[:200] + "...",
                "metadata_col": doc.metadata_col,
                "similarity_score": score,
                "file_type": doc.file_type,
                "created_at": doc.created_at
            })
    
    return unique_results[:10]  # Return top 10 results