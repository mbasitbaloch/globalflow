from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base


class Translation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    # tenant_id = Column(Integer, nullable=True)
    original_text = Column(JSONB, nullable=False)
    translated_text = Column(JSONB, nullable=False)
    target_lang = Column(String(10), nullable=False)
    content_type = Column(String(50), default="json")
    translated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)