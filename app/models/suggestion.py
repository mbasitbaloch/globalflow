import uuid
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, ForeignKey, DateTime, JSON, Float, Boolean, Text, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


# class Suggestion(Base):
#     __tablename__ = "suggestions"
#     id = Column(Integer, primary_key=True, index=True)
#     tenant_id = Column(String(128), nullable=False, index=True)
#     doc_type = Column(String(64), nullable=False)
#     domain = Column(String(64), nullable=False)
#     target_language = Column(String(32), nullable=False)
#     source_language = Column(String(32), nullable=False)
#     translation_id = Column(Integer, nullable=True)
#     path = Column(String, nullable=True)
#     original_text = Column(Text, nullable=False)
#     suggestions = Column(JSON, nullable=False)  # list of suggestion objects
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# class SuggestionAudit(Base):
#     __tablename__ = "suggestion_audit"
#     id = Column(Integer, primary_key=True, index=True)
#     # optional FK to suggestions.id
#     suggestion_id = Column(Integer, nullable=True)
#     tenant_id = Column(String(128), nullable=False)
#     user_id = Column(String(128), nullable=True)
#     translation_id = Column(Integer, nullable=True)
#     path = Column(String, nullable=True)
#     suggestion_hash = Column(String(128), nullable=True)
#     # "accept" | "reject" | "apply_all"
#     action = Column(String(16), nullable=False)
#     suggestion_type = Column(String(32), nullable=True)
#     before = Column(Text, nullable=True)
#     after = Column(Text, nullable=True)
#     meta = Column(JSON, nullable=True)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())


class SuggestionSegment(Base):
    __tablename__ = "suggestion_segments"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    translation_id = Column(Integer, nullable=False, index=True)
    doc_type = Column(String(100), nullable=True)
    domain = Column(String(255), nullable=True)
    target_language = Column(String(16), nullable=True)
    source_language = Column(String(16), nullable=True)
    path = Column(String(1024), nullable=False)
    original_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    options = relationship(
        "SuggestionOption", back_populates="segment", cascade="all, delete-orphan")


class SuggestionOption(Base):
    __tablename__ = "suggestion_options"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    segment_id = Column(Integer, ForeignKey(
        "suggestion_segments.id", ondelete="CASCADE"), nullable=False, index=True)
    # grammar, fluency, style, idiom, etc.
    suggestion_type = Column(String(64), nullable=False)
    before = Column(Text, nullable=False)
    after = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    blocked = Column(Boolean, default=False)
    risk = Column(String(32), nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    segment = relationship("SuggestionSegment", back_populates="options")


class SuggestionAudit(Base):
    __tablename__ = "suggestion_audits"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(64), nullable=False)
    translation_id = Column(Integer, nullable=False)
    segment_id = Column(Integer, nullable=True, index=True)
    option_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    user_id = Column(String(64), nullable=False)
    path = Column(String(1024), nullable=True)
    suggestion_hash = Column(String(128), nullable=False)
    action = Column(String(16), nullable=False)  # accept/reject
    before = Column(Text, nullable=True)
    after = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
