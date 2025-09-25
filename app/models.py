from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base


class Translation(Base):
    __tablename__ = "translation"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    shop_domain = Column(String, nullable=False)
    brand_tone = Column(String, nullable=False)
    original_text = Column(JSONB, nullable=False)
    translated_text = Column(JSONB, nullable=False)
    target_lang = Column(String(10), nullable=False)
    content_type = Column(String(50), default="json")
    translated_at = Column(DateTime(timezone=True),
                           server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        onupdate=func.now(), nullable=True)
