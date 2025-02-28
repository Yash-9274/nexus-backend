from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, JSON, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    metadata_col = Column(JSON, default={
        "summary": "",
        "keywords": [],
        "topics": [],
        "entities": [],
        "category": "Unknown"
    })
    embedding = Column(ARRAY(Float), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="documents") 