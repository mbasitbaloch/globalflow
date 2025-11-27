# app/models/suggestion.py
from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from ..database import Base


class Suggestion(Base):
    __tablename__ = "suggestions"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(128), nullable=False, index=True)
    doc_type = Column(String(64), nullable=False)
    domain = Column(String(64), nullable=False)
    language_pair = Column(String(32), nullable=False)
    translation_id = Column(Integer, nullable=True)
    path = Column(String, nullable=True)
    original_text = Column(Text, nullable=False)
    suggestions = Column(JSON, nullable=False)  # list of suggestion objects
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class SuggestionAudit(Base):
    __tablename__ = "suggestion_audit"
    id = Column(Integer, primary_key=True, index=True)
    # optional FK to suggestions.id
    suggestion_id = Column(Integer, nullable=True)
    tenant_id = Column(String(128), nullable=False)
    user_id = Column(String(128), nullable=True)
    translation_id = Column(Integer, nullable=True)
    path = Column(String, nullable=True)
    suggestion_hash = Column(String(128), nullable=True)
    # "accept" | "reject" | "apply_all"
    action = Column(String(16), nullable=False)
    suggestion_type = Column(String(32), nullable=True)
    before = Column(Text, nullable=True)
    after = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
