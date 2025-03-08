import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base

class AccessLevel(str, enum.Enum):
    VIEW = "VIEW"
    EDIT = "EDIT"

class ShareAccess(Base):
    __tablename__ = "share_access"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    shared_by_id = Column(Integer, ForeignKey("users.id"))
    shared_with_email = Column(String, nullable=False)
    access_level = Column(Enum(AccessLevel), default=AccessLevel.VIEW)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="shares")
    shared_by = relationship("User", back_populates="shared_documents") 